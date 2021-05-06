// Code in this file is a heavily modified version of the dynamic loader
// from android's bionic library. Here is the license for that project:

/*
 * Copyright (C) 2016 The Android Open Source Project
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#include <assert.h>
#include <dlfcn.h>
#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <atomic>
#include <cinttypes>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>
#include <limits.h>
#include <libgen.h>
#include <functional>
#define _GNU_SOURCE         /* See feature_test_macros(7) */
#include <link.h>

std::vector<std::string> split_path(const std::string& s, char delim) {
  const char* cur = s.c_str();
  const char* end = cur + s.size();
  if (cur == end) {
    return {};
  }
  std::vector<std::string> result;
  while (true) {
    // non-zero amount of chars
    const char* next = strchr(cur, delim);
    if (!next) {
      result.push_back(std::string(cur, end));
      break;
    }
    result.push_back(std::string(cur, next));
    cur = next + 1;
  }
  return result;
}


// https://stackoverflow.com/questions/23006930/the-shared-library-rpath-and-the-binary-rpath-priority/52647116#52647116
void replace_all(std::string& str, const std::string& from, const std::string& to) {
    if(from.empty())
        return;
    size_t start_pos = 0;
    while((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
    }
}

std::string resolve_path(const std::string& origin, const std::string& t) {
  std::string result = t;
  replace_all(result, "$ORIGIN", origin);
  char buf[PATH_MAX];
  char* resolved = realpath(result.c_str(), buf);
  if (!resolved) {
    return result;
  }
  return resolved;
}

std::string resolve_origin(const std::string& so_name) {
  char origin[PATH_MAX];
  realpath(so_name.c_str(), origin);
  dirname(origin);
  return origin;
}

template <typename... Args>
std::string stringf(const char* format, Args... args) {
  int size_s = snprintf(nullptr, 0, format, args...);
  std::string result(size_s + 1, 0);
  snprintf((char*)result.data(), size_s + 1, format, args...);
  return result;
}

// from bionic

// Get PAGE_SIZE and PAGE_MASK.
#include <sys/user.h>

// Returns the address of the page containing address 'x'.
#define PAGE_START(x) ((x)&PAGE_MASK)

// Returns the offset of address 'x' in its page.
#define PAGE_OFFSET(x) ((x) & ~PAGE_MASK)

// Returns the address of the next page after address 'x', unless 'x' is
// itself at the start of a page.
#define PAGE_END(x) PAGE_START((x) + (PAGE_SIZE - 1))

// from bionic
size_t phdr_table_get_load_size(
    const Elf64_Phdr* phdr_table,
    size_t phdr_count,
    Elf64_Addr* out_min_vaddr,
    Elf64_Addr* out_max_vaddr) {
  Elf64_Addr min_vaddr = UINTPTR_MAX;
  Elf64_Addr max_vaddr = 0;

  bool found_pt_load = false;
  for (size_t i = 0; i < phdr_count; ++i) {
    const Elf64_Phdr* phdr = &phdr_table[i];

    if (phdr->p_type != PT_LOAD) {
      continue;
    }
    found_pt_load = true;

    if (phdr->p_vaddr < min_vaddr) {
      min_vaddr = phdr->p_vaddr;
    }

    if (phdr->p_vaddr + phdr->p_memsz > max_vaddr) {
      max_vaddr = phdr->p_vaddr + phdr->p_memsz;
    }
  }
  if (!found_pt_load) {
    min_vaddr = 0;
  }

  min_vaddr = PAGE_START(min_vaddr);
  max_vaddr = PAGE_END(max_vaddr);

  if (out_min_vaddr != nullptr) {
    *out_min_vaddr = min_vaddr;
  }
  if (out_max_vaddr != nullptr) {
    *out_max_vaddr = max_vaddr;
  }
  return max_vaddr - min_vaddr;
}

#define MAYBE_MAP_FLAG(x, from, to) (((x) & (from)) ? (to) : 0)
#define PFLAGS_TO_PROT(x)                 \
  (MAYBE_MAP_FLAG((x), PF_X, PROT_EXEC) | \
   MAYBE_MAP_FLAG((x), PF_R, PROT_READ) | \
   MAYBE_MAP_FLAG((x), PF_W, PROT_WRITE))

struct GnuHash {
  GnuHash(const char* name) {
    uint32_t h = 5381;
    const uint8_t* name_bytes = reinterpret_cast<const uint8_t*>(name);
#pragma unroll 8
    while (*name_bytes != 0) {
      h += (h << 5) +
          *name_bytes++; // h*33 + c = h + h * 32 + c = h + h << 5 + c
    }
    hash = h;
    name_len = reinterpret_cast<const char*>(name_bytes) - name;
  }
  uint32_t hash;
  uint32_t name_len;
};

// end bionic

static void check_errno(bool success) {
  if (!success) {
    throw std::runtime_error(strerror(errno));
  }
}

extern "C" void __register_frame(void*);

struct MemFile {
  MemFile(const char* filename_) : mem_(nullptr), fd_(0), n_bytes_(0) {
    fd_ = open(filename_, O_RDONLY);
    check_errno(fd_ != -1);
    struct stat s;
    check_errno(-1 != fstat(fd_, &s));
    n_bytes_ = s.st_size;
    mem_ = mmap(nullptr, n_bytes_, PROT_READ, MAP_SHARED, fd_, 0);
    check_errno(MAP_FAILED != mem_);
  }
  MemFile(const MemFile&) = delete;
  const char* data() const {
    return (const char*)mem_;
  }
  ~MemFile() {
    if (mem_) {
      check_errno(0 == munmap((void*)mem_, n_bytes_));
    }
    if (fd_) {
      check_errno(0 == close(fd_));
    }
  }
  size_t size() {
    return n_bytes_;
  }
  int fd() const {
    return fd_;
  }

 private:
  int fd_;
  void* mem_;
  size_t n_bytes_;
};
template <typename... Args>
void error(const char* format, Args... args) {
  throw std::runtime_error(stringf(format, args...));
}

typedef void (*linker_dtor_function_t)();
typedef void (*linker_ctor_function_t)(int, char**, char**);

// https://refspecs.linuxfoundation.org/LSB_2.1.0/LSB-Core-generic/LSB-Core-generic/ehframehdr.html
struct EH_Frame_HDR {
  char version;
  char eh_frame_ptr_enc;
  char fde_count_enc;
  char table_enc;
  int32_t eh_frame_ptr;
};

typedef enum {
  JIT_NOACTION = 0,
  JIT_REGISTER_FN,
  JIT_UNREGISTER_FN
} jit_actions_t;

struct jit_code_entry {
  struct jit_code_entry* next_entry;
  struct jit_code_entry* prev_entry;
  const char* symfile_addr;
  uint64_t symfile_size;
};

struct jit_descriptor {
  uint32_t version;
  /* This type should be jit_actions_t, but we use uint32_t
     to be explicit about the bitwidth.  */
  uint32_t action_flag;
  struct jit_code_entry* relevant_entry;
  struct jit_code_entry* first_entry;
};


extern "C" void* __tls_get_addr(void*);

struct TLSEntry {
  size_t module_id;
  size_t offset;
};

struct TLSSegment {
  TLSSegment() {
    int r = pthread_key_create(&tls_key_, free);
    assert(r == 0);
  }
  void* addr(size_t offset) {
    if (mem_size_ == 0) {
      // this was a real TLS entry, fall back to the libc implementation
      TLSEntry real_get_addr = {module_id_, offset};
      // std::cout << "I THINK THE ID IS: " << module_id_ << " " << offset << "\n";
      return __tls_get_addr(&real_get_addr);
    } else {
      // this was a TLS entry for one of our modules, so we use pthreads to emulate
      // thread local state.
      void* start = pthread_getspecific(tls_key_);
      if (!start) {
        start = malloc(mem_size_);
        memcpy(start, initalization_image_, file_size_);
        pthread_setspecific(tls_key_, start);
      }
      return (void*)((const char*)start + offset);
    }
  }

 private:
  pthread_key_t tls_key_;
  void* initalization_image_;

  union {
    size_t file_size_;
    size_t module_id_;
  };
  size_t mem_size_;
  friend struct ElfFile;
};

struct TLSIndex {
  TLSSegment* segment;
  size_t offset;
};

static void* local__tls_get_addr(TLSIndex* idx) {
  return idx->segment->addr(idx->offset);
}

/* GDB puts a breakpoint in this function.  */
void __attribute__((noinline)) __jit_debug_register_code(){};

/* Make sure to specify the version statically, because the
   debugger may check the version before we can set it.  */
struct jit_descriptor __jit_debug_descriptor = {1, 0, 0, 0};

struct SystemLibrary {
  SystemLibrary() : handle_(RTLD_DEFAULT) {}
  SystemLibrary(void* handle) : handle_(handle) {}
  SystemLibrary(
      const char* library_name,
      std::vector<std::string>& search_path) {
    if (strchr(library_name, '/') == nullptr) {
      for (const std::string& path : search_path) {
        std::stringstream ss;
        ss << path << "/" << library_name;
        handle_ = dlopen(ss.str().c_str(), RTLD_LAZY | RTLD_LOCAL);
        if (handle_) {
          return;
        }
      }
    }
    // we failed in the search_path, or this is an absolute path
    handle_ = dlopen(library_name, RTLD_LAZY | RTLD_LOCAL);
    if (!handle_) {
      error("%s: %s", library_name, dlerror());
    }
  }
  SystemLibrary(const SystemLibrary& rhs) = delete;
  void* sym(const char* name) {
    return dlsym(handle_, name);
  }
  std::string last_error() const {
    return dlerror();
  }
  ~SystemLibrary() {
    if (handle_ && handle_ != RTLD_DEFAULT) {
      dlclose(handle_);
    }
  }

 private:
  void* handle_;
};

// TODO: dedup with ElfFile
struct AlreadyLoadedSymTable {
  AlreadyLoadedSymTable(const char* name, Elf64_Addr load_bias, const Elf64_Phdr* program_headers, size_t n_program_headers)
  : load_bias_(load_bias) {
    Elf64_Dyn* dynamic = nullptr;
    for (size_t i = 0; i < n_program_headers; ++i) {
      const Elf64_Phdr* phdr = &program_headers[i];

      // Segment addresses in memory.
      Elf64_Addr seg_start = phdr->p_vaddr + load_bias_;
      if (phdr->p_type == PT_DYNAMIC) {
        dynamic = reinterpret_cast<Elf64_Dyn*>(seg_start);
        break;
      }
    }

    if(!dynamic) {
      error("%s couldn't find PT_DYNAMIC", name, load_bias);
    }

    for (const Elf64_Dyn* d = dynamic; d->d_tag != DT_NULL; ++d) {
      void* addr = d->d_un.d_ptr > load_bias_ ? reinterpret_cast<void*>(d->d_un.d_ptr) : reinterpret_cast<void*>(load_bias_ + d->d_un.d_ptr);
      auto value = d->d_un.d_val;
      switch (d->d_tag) {
        case DT_SYMTAB:
          symtab_ = (Elf64_Sym*)addr;
          break;
        case DT_STRTAB:
          strtab_ = (const char*)addr;
          break;
        case DT_STRSZ:
          strtab_size_ = value;
          break;

        case DT_GNU_HASH: {
          gnu_nbucket_ = reinterpret_cast<uint32_t*>(addr)[0];
          // skip symndx
          gnu_maskwords_ = reinterpret_cast<uint32_t*>(addr)[2];
          gnu_shift2_ = reinterpret_cast<uint32_t*>(addr)[3];
          gnu_bloom_filter_ =
              reinterpret_cast<Elf64_Addr*>((Elf64_Addr)addr + 16);
          gnu_bucket_ =
              reinterpret_cast<uint32_t*>(gnu_bloom_filter_ + gnu_maskwords_);
          // amend chain for symndx = header[1]
          gnu_chain_ =
              gnu_bucket_ + gnu_nbucket_ - reinterpret_cast<uint32_t*>(addr)[1];
          --gnu_maskwords_;
        } break;
      }
    }

    if (!gnu_bucket_) {
      std::cout << name << ": no DT_GNU_HASH section, I don't know how to read DT_HASH...\n";
    }
  }
  void* sym(const char* name, GnuHash* precomputed_hash = nullptr) {
    if (!gnu_bucket_) {
      return nullptr;
    }
    GnuHash hash_obj = precomputed_hash ? *precomputed_hash : GnuHash(name);
    auto hash = hash_obj.hash;
    auto name_len = hash_obj.name_len;
    constexpr uint32_t kBloomMaskBits = sizeof(Elf64_Addr) * 8;

    const uint32_t word_num = (hash / kBloomMaskBits) & gnu_maskwords_;
    const Elf64_Addr bloom_word = gnu_bloom_filter_[word_num];
    const uint32_t h1 = hash % kBloomMaskBits;
    const uint32_t h2 = (hash >> gnu_shift2_) % kBloomMaskBits;

    if ((1 & (bloom_word >> h1) & (bloom_word >> h2)) != 1) {
      return nullptr;
    }

    uint32_t sym_idx = gnu_bucket_[hash % gnu_nbucket_];
    if (sym_idx == 0) {
      return nullptr;
    }

    uint32_t chain_value = 0;
    const Elf64_Sym* sym = nullptr;

    do {
      sym = symtab_ + sym_idx;
      chain_value = gnu_chain_[sym_idx];
      if ((chain_value >> 1) == (hash >> 1)) {
        if (static_cast<size_t>(sym->st_name) + name_len + 1 <= strtab_size_ &&
            memcmp(strtab_ + sym->st_name, name, name_len + 1) == 0 &&
            (ELF64_ST_BIND(sym->st_info) == STB_GLOBAL ||
             ELF64_ST_BIND(sym->st_info) == STB_WEAK)) {
          if (ELF64_ST_TYPE(sym->st_info) == STT_TLS) {
            return (void*) sym->st_value;
          } else {
            return (void*)(load_bias_ + sym->st_value);
          }
        }
      }
      ++sym_idx;
    } while ((chain_value & 1) == 0);
    return nullptr;
  }
private:
  Elf64_Addr load_bias_;
  size_t gnu_nbucket_;
  uint32_t* gnu_bucket_ = nullptr;
  uint32_t* gnu_chain_;
  uint32_t gnu_maskwords_;
  uint32_t gnu_shift2_;
  Elf64_Addr* gnu_bloom_filter_;
  const Elf64_Sym* symtab_;
  const char* strtab_;
  size_t strtab_size_;
};


static int iterate_cb(struct dl_phdr_info * info, size_t size, void * data) {
  auto fn = (std::function<int(struct dl_phdr_info * info, size_t size)>*)data;
  return (*fn)(info, size);
}

bool slow_find_tls_symbol_offset(const char* sym_name, TLSEntry* result) {
    bool found = false;
    std::function<int(struct dl_phdr_info *,size_t)> cb = [&](struct dl_phdr_info * info, size_t size) {
    // std::cout << "SEARCHING .. " << info->dlpi_name << "\n";
    AlreadyLoadedSymTable symtable(info->dlpi_name, info->dlpi_addr, info->dlpi_phdr, info->dlpi_phnum);
    void* sym_addr = symtable.sym(sym_name);
    if (sym_addr) {
      // std::cout << "FOUND IT IN: " << info->dlpi_name << " it has modid: " << info->dlpi_tls_modid << "\n";
      result->module_id = info->dlpi_tls_modid;
      result->offset = (Elf64_Addr) sym_addr;
      found = true;
      return 1;
    }
    return 0;
  };

  dl_iterate_phdr(iterate_cb, (void*)&cb);
  return found;
}


struct ElfFile {
  ElfFile(const ElfFile&) = delete;
  ElfFile(const char* filename, int argc, char** argv)
      : contents_(filename),
        mapped_library_(nullptr),
        name_(filename),
        argc_(argc),
        argv_(argv) {
    data_ = contents_.data();
    header_ = (Elf64_Ehdr*)data_;
    program_headers_ = (Elf64_Phdr*)(data_ + header_->e_phoff);
    n_program_headers_ = header_->e_phnum;
    // system library search path starts with the process global symbols.
    system_libraries_.push_back(std::make_unique<SystemLibrary>());
  }
  void add_system_library(void* handle) {
    system_libraries_.push_back(std::make_unique<SystemLibrary>(handle));
  }
  void reserve_address_space() {
    Elf64_Addr min_vaddr, max_vaddr;
    mapped_size_ = phdr_table_get_load_size(
        program_headers_, n_program_headers_, &min_vaddr, &max_vaddr);
    mapped_library_ = mmap(
        nullptr, mapped_size_, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    load_bias_ =
        (const char*)mapped_library_ - reinterpret_cast<const char*>(min_vaddr);
  }

  void load_segments() {
    // from bionic
    for (size_t i = 0; i < n_program_headers_; ++i) {
      const Elf64_Phdr* phdr = &program_headers_[i];

      // Segment addresses in memory.
      Elf64_Addr seg_start = phdr->p_vaddr + load_bias_;
      Elf64_Addr seg_end = seg_start + phdr->p_memsz;

      switch (phdr->p_type) {
        case PT_DYNAMIC:
          dynamic_ = reinterpret_cast<Elf64_Dyn*>(seg_start);
          break;
        case PT_GNU_EH_FRAME:
          eh_frame_hdr_ = reinterpret_cast<EH_Frame_HDR*>(seg_start);
          assert(eh_frame_hdr_->eh_frame_ptr_enc == 0x1b);
          eh_frame_ =
              (void*)((int64_t)&eh_frame_hdr_->eh_frame_ptr + eh_frame_hdr_->eh_frame_ptr);
          break;
        case PT_TLS:
          tls_.file_size_ = phdr->p_filesz;
          tls_.mem_size_ = phdr->p_memsz;
          tls_.initalization_image_ = (void*)seg_start;
          break;
      };

      if (phdr->p_type != PT_LOAD) {
        continue;
      }

      Elf64_Addr seg_page_start = PAGE_START(seg_start);
      Elf64_Addr seg_page_end = PAGE_END(seg_end);

      Elf64_Addr seg_file_end = seg_start + phdr->p_filesz;

      // File offsets.
      Elf64_Addr file_start = phdr->p_offset;
      Elf64_Addr file_end = file_start + phdr->p_filesz;

      Elf64_Addr file_page_start = PAGE_START(file_start);
      Elf64_Addr file_length = file_end - file_page_start;

      if (contents_.size() <= 0) {
        error(
            "\"%s\" invalid file size: %" PRId64,
            name_.c_str(),
            contents_.size());
      }

      if (file_end > contents_.size()) {
        error(
            "invalid ELF file \"%s\" load segment[%zd]:"
            " p_offset (%p) + p_filesz (%p) ( = %p) past end of file "
            "(0x%" PRIx64 ")",
            name_.c_str(),
            i,
            reinterpret_cast<void*>(phdr->p_offset),
            reinterpret_cast<void*>(phdr->p_filesz),
            reinterpret_cast<void*>(file_end),
            contents_.size());
      }

      if (file_length != 0) {
        int prot = PFLAGS_TO_PROT(phdr->p_flags);

        void* seg_addr = mmap64(
            reinterpret_cast<void*>(seg_page_start),
            file_length,
            prot,
            MAP_FIXED | MAP_PRIVATE,
            contents_.fd(),
            file_page_start);
        if (seg_addr == MAP_FAILED) {
          error(
              "couldn't map \"%s\" segment %zd: %s",
              name_.c_str(),
              i,
              strerror(errno));
        }
      }

      // if the segment is writable, and does not end on a page boundary,
      // zero-fill it until the page limit.
      if ((phdr->p_flags & PF_W) != 0 && PAGE_OFFSET(seg_file_end) > 0) {
        memset(
            reinterpret_cast<void*>(seg_file_end),
            0,
            PAGE_SIZE - PAGE_OFFSET(seg_file_end));
      }

      seg_file_end = PAGE_END(seg_file_end);

      // seg_file_end is now the first page address after the file
      // content. If seg_end is larger, we need to zero anything
      // between them. This is done by using a private anonymous
      // map for all extra pages.
      if (seg_page_end > seg_file_end) {
        size_t zeromap_size = seg_page_end - seg_file_end;
        void* zeromap = mmap(
            reinterpret_cast<void*>(seg_file_end),
            zeromap_size,
            PFLAGS_TO_PROT(phdr->p_flags),
            MAP_FIXED | MAP_ANONYMOUS | MAP_PRIVATE,
            -1,
            0);
        if (zeromap == MAP_FAILED) {
          error(
              "couldn't zero fill \"%s\" gap: %s",
              name_.c_str(),
              strerror(errno));
        }
      }
    }
  }
  const char* get_string(int idx) {
    return strtab_ + idx;
  }

  void read_dynamic_section() {
    for (const Elf64_Dyn* d = dynamic_; d->d_tag != DT_NULL; ++d) {
      void* addr = reinterpret_cast<void*>(load_bias_ + d->d_un.d_ptr);
      auto value = d->d_un.d_val;
      switch (d->d_tag) {
        case DT_SYMTAB:
          symtab_ = (Elf64_Sym*)addr;
          break;
        case DT_STRTAB:
          strtab_ = (const char*)addr;
          break;

        case DT_STRSZ:
          strtab_size_ = value;
          break;

        case DT_JMPREL:
          plt_rela_ = (Elf64_Rela*)addr;
          break;
        case DT_PLTRELSZ:
          n_plt_rela_ = value / sizeof(Elf64_Rela);
          break;
        case DT_RELA:
          rela_ = (Elf64_Rela*)addr;
          break;
        case DT_RELASZ:
          n_rela_ = value / sizeof(Elf64_Rela);
          break;

        case DT_INIT:
          init_func_ = reinterpret_cast<linker_ctor_function_t>(
              load_bias_ + d->d_un.d_ptr);
          break;

        case DT_FINI:
          fini_func_ = reinterpret_cast<linker_dtor_function_t>(
              load_bias_ + d->d_un.d_ptr);
          break;

        case DT_INIT_ARRAY:
          init_array_ = reinterpret_cast<linker_ctor_function_t*>(
              load_bias_ + d->d_un.d_ptr);
          break;

        case DT_INIT_ARRAYSZ:
          n_init_array_ =
              static_cast<uint32_t>(d->d_un.d_val) / sizeof(Elf64_Addr);
          break;

        case DT_FINI_ARRAY:
          fini_array_ = reinterpret_cast<linker_dtor_function_t*>(
              load_bias_ + d->d_un.d_ptr);
          break;

        case DT_FINI_ARRAYSZ:
          n_fini_array_ =
              static_cast<uint32_t>(d->d_un.d_val) / sizeof(Elf64_Addr);
          break;

        case DT_HASH:
          break;

        case DT_GNU_HASH: {
          gnu_nbucket_ = reinterpret_cast<uint32_t*>(addr)[0];
          // skip symndx
          gnu_maskwords_ = reinterpret_cast<uint32_t*>(addr)[2];
          gnu_shift2_ = reinterpret_cast<uint32_t*>(addr)[3];
          gnu_bloom_filter_ =
              reinterpret_cast<Elf64_Addr*>((Elf64_Addr)addr + 16);
          gnu_bucket_ =
              reinterpret_cast<uint32_t*>(gnu_bloom_filter_ + gnu_maskwords_);
          // amend chain for symndx = header[1]
          gnu_chain_ =
              gnu_bucket_ + gnu_nbucket_ - reinterpret_cast<uint32_t*>(addr)[1];
          --gnu_maskwords_;
        } break;
      }
    }

    if (!gnu_bucket_) {
      error("%s: no DT_GNU_HASH section, I don't know how to read DT_HASH...", name_.c_str());
    }

    // pass 2 for things that require the strtab_ to be loaded
    std::vector<const char*> needed;
    std::string runpath = "";
    for (const Elf64_Dyn* d = dynamic_; d->d_tag != DT_NULL; ++d) {
      switch (d->d_tag) {
        case DT_NEEDED:
          needed.push_back(get_string(d->d_un.d_val));
          break;
        case DT_RPATH: /* not quite correct, because this is a different order than runpath,
                          but better than not processing it at all */
        case DT_RUNPATH:
          runpath = get_string(d->d_un.d_val);
          break;
      }
    }
    resolve_needed_libraries(runpath, needed);
  }

  void resolve_needed_libraries(
      const std::string& runpath,
      const std::vector<const char*>& needed) {

    std::string origin = resolve_origin(name_);
    std::vector<std::string> paths = split_path(runpath, ':');
    for (size_t i = 0; i < paths.size(); ++i) {
      paths[i] = resolve_path(origin, paths[i]);
    }

    for (const char* name : needed) {
      if (strcmp(name, "libtorch_python.so") == 0) {
        // torchvision expects it...
        continue;
      }
      system_libraries_.push_back(std::make_unique<SystemLibrary>(name, paths));
    }
  }

  void* sym(const char* name, GnuHash* precomputed_hash = nullptr) {
    GnuHash hash_obj = precomputed_hash ? *precomputed_hash : GnuHash(name);
    auto hash = hash_obj.hash;
    auto name_len = hash_obj.name_len;
    constexpr uint32_t kBloomMaskBits = sizeof(Elf64_Addr) * 8;

    const uint32_t word_num = (hash / kBloomMaskBits) & gnu_maskwords_;
    const Elf64_Addr bloom_word = gnu_bloom_filter_[word_num];
    const uint32_t h1 = hash % kBloomMaskBits;
    const uint32_t h2 = (hash >> gnu_shift2_) % kBloomMaskBits;

    if ((1 & (bloom_word >> h1) & (bloom_word >> h2)) != 1) {
      return nullptr;
    }

    uint32_t sym_idx = gnu_bucket_[hash % gnu_nbucket_];
    if (sym_idx == 0) {
      return nullptr;
    }

    uint32_t chain_value = 0;
    const Elf64_Sym* sym = nullptr;

    do {
      sym = symtab_ + sym_idx;
      chain_value = gnu_chain_[sym_idx];
      if ((chain_value >> 1) == (hash >> 1)) {
        if (static_cast<size_t>(sym->st_name) + name_len + 1 <= strtab_size_ &&
            memcmp(strtab_ + sym->st_name, name, name_len + 1) == 0 &&
            (ELF64_ST_BIND(sym->st_info) == STB_GLOBAL ||
             ELF64_ST_BIND(sym->st_info) == STB_WEAK)) {
          if (ELF64_ST_TYPE(sym->st_info) == STT_TLS) {
            return (void*) sym->st_value;
          } else {
            return (void*)(load_bias_ + sym->st_value);
          }
        }
      }
      ++sym_idx;
    } while ((chain_value & 1) == 0);
    return nullptr;
  }

  Elf64_Addr lookup_symbol(const char* name, bool must_be_defined) {
    for (const auto& sys_lib : system_libraries_) {
      void* r = sys_lib->sym(name);
      if (r) {
        return (Elf64_Addr)r;
      }
    }
    void* r = sym(name);
    if (!r && must_be_defined) {
      error("%s: symbol not found in ElfFile lookup", name);
    }
    return (Elf64_Addr)r;
  }
  void relocate_one(const Elf64_Rela& reloc) {
    void* const rel_target =
        reinterpret_cast<void*>(reloc.r_offset + load_bias_);
    const uint32_t r_type = ELF64_R_TYPE(reloc.r_info);
    const uint32_t r_sym = ELF64_R_SYM(reloc.r_info);

    const char* sym_name = nullptr;
    Elf64_Addr sym_addr = 0;

    if (r_type == 0) {
      return;
    }
    if (r_sym != 0) {
      auto sym_st = symtab_[r_sym];
      sym_name = get_string(sym_st.st_name);
      // std::cout << "PROCESSING SYMBOL: " << sym_name << "\n";

      bool must_be_defined =
          ELF64_ST_BIND(sym_st.st_info) != STB_WEAK || sym_st.st_shndx != 0;
      if (r_type == R_X86_64_JUMP_SLOT &&
          strcmp(sym_name, "__tls_get_addr") == 0) {
        sym_addr = (Elf64_Addr)local__tls_get_addr;
      } else {
        sym_addr = lookup_symbol(sym_name, must_be_defined);
      }
      if (sym_addr == 0) {
        // std::cout << "SKIPPING RELOCATION FOR WEAK THING: " << sym_name <<
        // "\n";
        return;
      }
      // if ((r_type == R_X86_64_DTPOFF64 || r_type == R_X86_64_DTPMOD64) &&
      //     (Elf64_Addr)sym(sym_name) != sym_addr) {
      //   error(
      //       "cannot resolve thread_local symbol to another module: %s",
      //       sym_name);
      // }
    }

    switch (r_type) {
      case R_X86_64_JUMP_SLOT:
      case R_X86_64_64:
      case R_X86_64_GLOB_DAT: {
        const Elf64_Addr result = sym_addr + reloc.r_addend;
        *static_cast<Elf64_Addr*>(rel_target) = result;
      } break;
      case R_X86_64_RELATIVE: {
        // In practice, r_sym is always zero, but if it weren't, the linker
        // would still look up the referenced symbol (and abort if the symbol
        // isn't found), even though it isn't used.
        const Elf64_Addr result = load_bias_ + reloc.r_addend;
        *static_cast<Elf64_Addr*>(rel_target) = result;
      } break;
      case R_X86_64_32: {
        const Elf32_Addr result = sym_addr + reloc.r_addend;
        *static_cast<Elf32_Addr*>(rel_target) = result;
      } break;
      case R_X86_64_PC32: {
        const Elf64_Addr target = sym_addr + reloc.r_addend;
        const Elf64_Addr base = reinterpret_cast<Elf64_Addr>(rel_target);
        const Elf32_Addr result = target - base;
        *static_cast<Elf32_Addr*>(rel_target) = result;
      } break;
      case R_X86_64_DTPMOD64: {
        if (sym_addr != 0 && (Elf64_Addr)sym(sym_name) != sym_addr) {
          TLSEntry entry;
          if (!slow_find_tls_symbol_offset(sym_name, &entry)) {
            error("%s: FAILED TO FIND TLS ENTRY", sym_name);
          }
          auto seg = new TLSSegment();
          seg->mem_size_ = 0;
          seg->module_id_ = entry.module_id;
          *static_cast<TLSSegment**>(rel_target) = seg;
        } else {
          *static_cast<TLSSegment**>(rel_target) = &tls_;
        }
      } break;
      case R_X86_64_DTPOFF64: {
        if (sym_addr != 0 && (Elf64_Addr)sym(sym_name) != sym_addr) {
          TLSEntry entry;
          if (!slow_find_tls_symbol_offset(sym_name, &entry)) {
            error("%s: FAILED TO FIND TLS ENTRY", sym_name);
          }
          *static_cast<Elf64_Addr*>(rel_target) = entry.offset + reloc.r_addend;
        } else {
          const Elf64_Addr result = sym_addr + reloc.r_addend;
          *static_cast<Elf64_Addr*>(rel_target) = result;
        }
      } break;
      default:
        error("unknown reloc type %d in \"%s\"", r_type, name_.c_str());
        break;
    }
  }
  void relocate() {
    for (size_t i = 0; i < n_rela_; ++i) {
      relocate_one(rela_[i]);
    }
    for (size_t i = 0; i < n_plt_rela_; ++i) {
      relocate_one(plt_rela_[i]);
    }
  }

  void initialize() {
    call_function(init_func_);
    for (size_t i = 0; i < n_init_array_; ++i) {
      call_function(init_array_[i]);
    }
    initialized_ = true;
  }

  void finalize() {
    for (size_t i = n_fini_array_; i > 0; --i) {
      call_function(fini_array_[i - 1]);
    }
    call_function(fini_func_);
  }

  void load() {
    reserve_address_space();
    load_segments();
    read_dynamic_section();
    relocate();
    __register_frame(eh_frame_);
    // gdb_entry_.next_entry = nullptr;
    // gdb_entry_.prev_entry = nullptr;
    // gdb_entry_.symfile_addr = contents_.data();
    // gdb_entry_.symfile_size = contents_.size();
    // __jit_debug_descriptor.action_flag = JIT_REGISTER_FN;
    // __jit_debug_descriptor.first_entry = &gdb_entry_;
    // __jit_debug_descriptor.relevant_entry = &gdb_entry_;
    // __jit_debug_register_code();
    initialize();
    std::cout << "target modules add " << name_.c_str() << "\n";
    std::cout << "target modules load -f " << name_.c_str() << " -s "
              << std::hex << "0x" << load_bias_ << "\n";
  }

  ~ElfFile() {
    if (initialized_) {
      finalize();
    }
    if (mapped_library_) {
      check_errno(0 == munmap(mapped_library_, mapped_size_));
    }
  }
  void call_function(linker_dtor_function_t f) {
    if (f == nullptr || (int64_t)f == -1)
      return;
    f();
  }
  void call_function(linker_ctor_function_t f) {
    if (f == nullptr || (int64_t)f == -1)
      return;
    f(argc_, argv_, environ);
  }

 private:
  const char* data_;
  const Elf64_Ehdr* header_;
  const Elf64_Phdr* program_headers_;
  const Elf64_Dyn* dynamic_;
  const EH_Frame_HDR* eh_frame_hdr_;
  void* eh_frame_;
  size_t n_program_headers_;
  void* mapped_library_;
  size_t mapped_size_;
  Elf64_Addr load_bias_;
  MemFile contents_;
  std::string name_;
  const Elf64_Sym* symtab_;
  const char* strtab_;
  Elf64_Rela* plt_rela_;
  size_t n_plt_rela_;
  Elf64_Rela* rela_;
  size_t n_rela_;
  linker_ctor_function_t init_func_;
  linker_ctor_function_t* init_array_;
  linker_dtor_function_t fini_func_;
  linker_dtor_function_t* fini_array_;
  size_t n_init_array_;
  size_t n_fini_array_;
  size_t strtab_size_;
  size_t gnu_nbucket_;
  uint32_t* gnu_bucket_ = nullptr;
  uint32_t* gnu_chain_;
  uint32_t gnu_maskwords_;
  uint32_t gnu_shift2_;
  Elf64_Addr* gnu_bloom_filter_;

  int argc_;
  char** argv_;
  bool initialized_ = false;

  TLSSegment tls_;

  jit_code_entry gdb_entry_;

  std::vector<std::unique_ptr<SystemLibrary>> system_libraries_;
};

using func_t = int (*)(int, int);

void printit(const std::vector<std::string>& strs) {
  std::cout << "{\n";
  for (const std::string& s : strs) {
    std::cout << s << "\n";
  }
  std::cout << "}\n";
}

std::vector<std::unique_ptr<ElfFile>> loaded_files_;

static void* deploy_self = nullptr;

__attribute__((visibility("default"))) extern "C" void set_deploy_self(
    void* v) {
  deploy_self = v;
}

typedef void (*dl_funcptr)(void);
extern "C" dl_funcptr _PyImport_FindSharedFuncptr(
    const char* prefix,
    const char* shortname,
    const char* pathname,
    FILE* fp) {
  char* args[] = {"deploy"};
  loaded_files_.emplace_back(std::make_unique<ElfFile>(pathname, 1, args));
  ElfFile& lib = *loaded_files_.back();
  lib.add_system_library(deploy_self);
  lib.load();
  std::stringstream ss;
  ss << prefix << "_" << shortname;
  auto r = (dl_funcptr)lib.sym(ss.str().c_str());
  assert(r);
  return r;
}
