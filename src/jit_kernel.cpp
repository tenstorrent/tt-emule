#include "tt_emule/jit_kernel.hpp"
#include <dlfcn.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tt_emule {

// TT_EMULE_JIT_INCLUDE_DIR is injected by CMake as a compile definition.
#ifndef TT_EMULE_JIT_INCLUDE_DIR
#error "TT_EMULE_JIT_INCLUDE_DIR must be defined by CMake (path to include/jit_hw)"
#endif

static std::string jit_include_dir_from_cmake() {
    return TT_EMULE_JIT_INCLUDE_DIR;
}

KernelFn jit_compile_kernel(const std::string& kernel_src_path,
                             const std::vector<uint32_t>& compile_args,
                             const std::string& jit_include_dir_override) {
    const std::string& jit_inc = jit_include_dir_override.empty()
                                 ? jit_include_dir_from_cmake()
                                 : jit_include_dir_override;

    // 1. Make sure the kernel source file exists
    if (!std::filesystem::exists(kernel_src_path)) {
        throw std::runtime_error("jit_compile_kernel: kernel source not found: "
                                 + kernel_src_path);
    }
    std::string abs_kernel = std::filesystem::absolute(kernel_src_path).string();

    // 2. Create a temp directory
    char tmpdir[] = "/tmp/tt_emule_jit_XXXXXX";
    if (!mkdtemp(tmpdir))
        throw std::runtime_error("jit_compile_kernel: mkdtemp failed");
    std::string dir(tmpdir);

    // 3. Write wrapper.cpp
    std::string wrapper_path = dir + "/wrapper.cpp";
    {
        std::ofstream f(wrapper_path);
        if (!f)
            throw std::runtime_error("jit_compile_kernel: cannot write " + wrapper_path);
        f << "#include \"jit_kernel_stubs.hpp\"\n";
        f << "#include \"" << abs_kernel << "\"\n";
        // Provide a C-linkage entry point so dlsym can find it without
        // knowing the C++ mangled name of kernel_main().
        f << "extern \"C\" { void __emule_kernel_entry() { kernel_main(); } }\n";
    }

    // 4. Build -DKERNEL_COMPILE_TIME_ARGS=... flag
    std::string ct_flag;
    if (!compile_args.empty()) {
        std::ostringstream ss;
        ss << "-DKERNEL_COMPILE_TIME_ARGS=";
        for (size_t i = 0; i < compile_args.size(); ++i) {
            if (i) ss << ',';
            ss << compile_args[i];
        }
        ct_flag = ss.str();
    }

    // parent of jit_hw/ gives us access to kernel_api/
    std::string parent_inc = std::filesystem::path(jit_inc).parent_path().string();

    // 5. Build and run compile command
    std::string so_path = dir + "/kernel.so";
    std::ostringstream cmd;
    cmd << "g++ -std=c++17 -fPIC -shared -O1"
        << " -I\"" << jit_inc << "\""
        << " -I\"" << parent_inc << "\""
        << " -o \"" << so_path << "\"";
    if (!ct_flag.empty())
        cmd << " \"" << ct_flag << "\"";
    cmd << " \"" << wrapper_path << "\"";
    cmd << " 2>&1";  // capture stderr

    std::string full_cmd = cmd.str();
    int rc = std::system(full_cmd.c_str());
    if (rc != 0) {
        throw std::runtime_error(
            "jit_compile_kernel: g++ failed (exit " + std::to_string(rc) +
            ") for kernel: " + kernel_src_path);
    }

    // 6. dlopen the shared object
    void* handle = dlopen(so_path.c_str(), RTLD_NOW);
    if (!handle) {
        throw std::runtime_error(
            std::string("jit_compile_kernel: dlopen failed: ") + dlerror());
    }

    // 7. Resolve the C-linkage entry point
    using RawFn = void (*)();
    dlerror(); // clear any previous error
    RawFn fn = reinterpret_cast<RawFn>(dlsym(handle, "__emule_kernel_entry"));
    const char* err = dlerror();
    if (err) {
        std::string msg(err); // copy before any other libc call can overwrite the buffer
        dlclose(handle);
        throw std::runtime_error(
            "jit_compile_kernel: dlsym(__emule_kernel_entry) failed: " + msg);
    }

    // 8. Return KernelFn; keep handle alive via shared_ptr with custom deleter
    auto shared_handle = std::shared_ptr<void>(handle, [](void* h) { dlclose(h); });
    return [fn, shared_handle]() {
        fn();
    };
}

} // namespace tt_emule
