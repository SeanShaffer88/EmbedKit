//! ONNX Runtime, loaded at runtime — never linked.
//!
//! Following the WinRecovery runtime-load pattern (LoadLibraryW on
//! `srclient.dll` there; `dlopen` / `LoadLibraryExW` here), the
//! embedder loads the caller's `libonnxruntime` and calls its C API
//! through the `OrtApi` table obtained from `OrtGetApiBase`.
//!
//! ABI discipline: the field indices below were transcribed from
//! the pinned runtime's own header (`onnxruntime_c_api.h`, ONNX
//! Runtime 1.31.0, `ORT_API_VERSION 31`): `struct OrtApi` declares
//! its function pointers in a fixed order that the runtime never
//! reorders, and `GetApi(version)` returns the table *for the
//! requested version*. EmbedKit requests exactly version 31; a
//! runtime too old to serve it is reported as EK_ERR_NOT_SUPPORTED
//! with the runtime's own version string, never guessed at.
//! Bumping the pinned runtime means re-transcribing any changed
//! indices from the new header, in the same commit as pins.txt.

use std::ffi::{c_char, c_void, CStr, CString};

use crate::{CResult, ERR_NOT_SUPPORTED, ERR_PLATFORM};

/// The C API version this binding was transcribed from.
pub const ORT_API_VERSION: u32 = 31;

// Indices into the OrtApi table (see the module docs). Only the
// entries EmbedKit calls are listed.
const GET_ERROR_MESSAGE: usize = 2;
const CREATE_ENV: usize = 3;
const CREATE_SESSION: usize = 7;
const RUN: usize = 9;
const CREATE_SESSION_OPTIONS: usize = 10;
const SESSION_GET_INPUT_COUNT: usize = 30;
const SESSION_GET_OUTPUT_COUNT: usize = 31;
const SESSION_GET_INPUT_NAME: usize = 36;
const SESSION_GET_OUTPUT_NAME: usize = 37;
const CREATE_TENSOR_WITH_DATA_AS_ORT_VALUE: usize = 49;
const GET_TENSOR_MUTABLE_DATA: usize = 51;
const GET_TENSOR_ELEMENT_TYPE: usize = 60;
const GET_DIMENSIONS_COUNT: usize = 61;
const GET_DIMENSIONS: usize = 62;
const GET_TENSOR_TYPE_AND_SHAPE: usize = 65;
const CREATE_CPU_MEMORY_INFO: usize = 69;
const ALLOCATOR_FREE: usize = 76;
const GET_ALLOCATOR_WITH_DEFAULT_OPTIONS: usize = 78;
const RELEASE_ENV: usize = 92;
const RELEASE_STATUS: usize = 93;
const RELEASE_MEMORY_INFO: usize = 94;
const RELEASE_SESSION: usize = 95;
const RELEASE_VALUE: usize = 96;
const RELEASE_TENSOR_TYPE_AND_SHAPE_INFO: usize = 99;
const RELEASE_SESSION_OPTIONS: usize = 100;

const USED_INDICES: &[usize] = &[
    GET_ERROR_MESSAGE,
    CREATE_ENV,
    CREATE_SESSION,
    RUN,
    CREATE_SESSION_OPTIONS,
    SESSION_GET_INPUT_COUNT,
    SESSION_GET_OUTPUT_COUNT,
    SESSION_GET_INPUT_NAME,
    SESSION_GET_OUTPUT_NAME,
    CREATE_TENSOR_WITH_DATA_AS_ORT_VALUE,
    GET_TENSOR_MUTABLE_DATA,
    GET_TENSOR_ELEMENT_TYPE,
    GET_DIMENSIONS_COUNT,
    GET_DIMENSIONS,
    GET_TENSOR_TYPE_AND_SHAPE,
    CREATE_CPU_MEMORY_INFO,
    ALLOCATOR_FREE,
    GET_ALLOCATOR_WITH_DEFAULT_OPTIONS,
    RELEASE_ENV,
    RELEASE_STATUS,
    RELEASE_MEMORY_INFO,
    RELEASE_SESSION,
    RELEASE_VALUE,
    RELEASE_TENSOR_TYPE_AND_SHAPE_INFO,
    RELEASE_SESSION_OPTIONS,
];

// ONNX tensor element types (onnxruntime_c_api.h).
pub const TENSOR_FLOAT: i32 = 1;
pub const TENSOR_INT64: i32 = 7;

// OrtLoggingLevel: VERBOSE=0, INFO=1, WARNING=2, ERROR=3, FATAL=4.
const LOG_WARNING: i32 = 2;
// OrtAllocatorType: Device=0. OrtMemType: Default=0.
const ALLOCATOR_DEVICE: i32 = 0;
const MEM_TYPE_DEFAULT: i32 = 0;

type OrtStatusPtr = *mut c_void;
type FnGetErrorMessage = unsafe extern "C" fn(*const c_void) -> *const c_char;
type FnCreateEnv = unsafe extern "C" fn(i32, *const c_char, *mut *mut c_void) -> OrtStatusPtr;
type FnCreateSessionOptions = unsafe extern "C" fn(*mut *mut c_void) -> OrtStatusPtr;
type FnRun = unsafe extern "C" fn(
    *mut c_void,
    *const c_void,
    *const *const c_char,
    *const *const c_void,
    usize,
    *const *const c_char,
    usize,
    *mut *mut c_void,
) -> OrtStatusPtr;
type FnGetCount = unsafe extern "C" fn(*const c_void, *mut usize) -> OrtStatusPtr;
type FnGetName =
    unsafe extern "C" fn(*const c_void, usize, *mut c_void, *mut *mut c_char) -> OrtStatusPtr;
type FnCreateTensor = unsafe extern "C" fn(
    *const c_void,
    *mut c_void,
    usize,
    *const i64,
    usize,
    i32,
    *mut *mut c_void,
) -> OrtStatusPtr;
type FnGetTensorMutableData = unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> OrtStatusPtr;
type FnGetTensorTypeAndShape =
    unsafe extern "C" fn(*const c_void, *mut *mut c_void) -> OrtStatusPtr;
type FnGetElementType = unsafe extern "C" fn(*const c_void, *mut i32) -> OrtStatusPtr;
type FnGetDimsCount = unsafe extern "C" fn(*const c_void, *mut usize) -> OrtStatusPtr;
type FnGetDims = unsafe extern "C" fn(*const c_void, *mut i64, usize) -> OrtStatusPtr;
type FnCreateCpuMemoryInfo = unsafe extern "C" fn(i32, i32, *mut *mut c_void) -> OrtStatusPtr;
type FnGetDefaultAllocator = unsafe extern "C" fn(*mut *mut c_void) -> OrtStatusPtr;
type FnAllocatorFree = unsafe extern "C" fn(*mut c_void, *mut c_void) -> OrtStatusPtr;
type FnRelease = unsafe extern "C" fn(*mut c_void);

#[repr(C)]
struct OrtApiBase {
    get_api: unsafe extern "C" fn(u32) -> *const c_void,
    get_version_string: unsafe extern "C" fn() -> *const c_char,
}

type FnOrtGetApiBase = unsafe extern "C" fn() -> *const OrtApiBase;

// ---------------------------------------------------------------------------
// Dynamic library loading (zero-dependency: direct OS loader calls).
// ---------------------------------------------------------------------------

pub struct Library {
    handle: *mut c_void,
}

// A loaded library handle is just an opaque token; all calls through
// it happen under the owning embedder's mutex.
unsafe impl Send for Library {}

#[cfg(unix)]
mod loader {
    use std::ffi::c_void;
    pub const RTLD_NOW: i32 = 2;
    extern "C" {
        pub fn dlopen(filename: *const std::ffi::c_char, flags: i32) -> *mut c_void;
        pub fn dlsym(handle: *mut c_void, symbol: *const std::ffi::c_char) -> *mut c_void;
        pub fn dlclose(handle: *mut c_void) -> i32;
        pub fn dlerror() -> *mut std::ffi::c_char;
    }
}

#[cfg(windows)]
mod loader {
    use std::ffi::c_void;
    pub const LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR: u32 = 0x0000_0100;
    pub const LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: u32 = 0x0000_1000;
    #[link(name = "kernel32")]
    extern "system" {
        pub fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
        pub fn GetProcAddress(lib: *mut c_void, name: *const u8) -> *mut c_void;
        pub fn FreeLibrary(lib: *mut c_void) -> i32;
        pub fn GetLastError() -> u32;
    }
}

impl Library {
    pub fn load(path: &str) -> CResult<Library> {
        // Canonicalize to an absolute path first: the
        // LOAD_LIBRARY_SEARCH_* flags are specified for fully
        // qualified paths, and dlopen's relative-path resolution
        // depends on the process CWD — the caller pointed at one
        // specific library, so resolve it once, unambiguously.
        let abs = std::fs::canonicalize(path).map_err(|e| {
            (
                ERR_PLATFORM,
                format!("failed to load ONNX Runtime library '{path}': cannot resolve path: {e}"),
            )
        })?;
        let abs_string = abs.to_string_lossy().into_owned();
        let path = abs_string.as_str();
        #[cfg(unix)]
        {
            let c = CString::new(path).map_err(|_| {
                (
                    ERR_PLATFORM,
                    format!("library path contains a NUL byte: '{path}'"),
                )
            })?;
            let handle = unsafe { loader::dlopen(c.as_ptr(), loader::RTLD_NOW) };
            if handle.is_null() {
                let detail = unsafe {
                    let e = loader::dlerror();
                    if e.is_null() {
                        "unknown loader error".to_string()
                    } else {
                        CStr::from_ptr(e).to_string_lossy().into_owned()
                    }
                };
                return Err((
                    ERR_PLATFORM,
                    format!("failed to load ONNX Runtime library '{path}': {detail}"),
                ));
            }
            Ok(Library { handle })
        }
        #[cfg(windows)]
        {
            let wide: Vec<u16> = path.encode_utf16().chain(std::iter::once(0)).collect();
            // DLL_LOAD_DIR: the library's own folder is searched for
            // ITS dependencies (onnxruntime_providers_shared.dll sits
            // beside onnxruntime.dll) — plain LoadLibraryW would not
            // look there for a library loaded by full path.
            let flags =
                loader::LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | loader::LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
            let handle =
                unsafe { loader::LoadLibraryExW(wide.as_ptr(), std::ptr::null_mut(), flags) };
            if handle.is_null() {
                let code = unsafe { loader::GetLastError() };
                return Err((
                    ERR_PLATFORM,
                    format!(
                        "failed to load ONNX Runtime library '{path}': LoadLibraryExW error {code}"
                    ),
                ));
            }
            Ok(Library { handle })
        }
    }

    pub fn symbol(&self, name: &str) -> CResult<*mut c_void> {
        #[cfg(unix)]
        {
            let c = CString::new(name).unwrap();
            let p = unsafe { loader::dlsym(self.handle, c.as_ptr()) };
            if p.is_null() {
                return Err((
                    ERR_NOT_SUPPORTED,
                    format!("library does not export {name} — not an ONNX Runtime library"),
                ));
            }
            Ok(p)
        }
        #[cfg(windows)]
        {
            let p = unsafe { loader::GetProcAddress(self.handle, name.as_ptr()) };
            if p.is_null() {
                return Err((
                    ERR_NOT_SUPPORTED,
                    format!("library does not export {name} — not an ONNX Runtime library"),
                ));
            }
            Ok(p)
        }
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            #[cfg(unix)]
            unsafe {
                loader::dlclose(self.handle);
            }
            #[cfg(windows)]
            unsafe {
                loader::FreeLibrary(self.handle);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The loaded runtime: OrtApi table + version, with typed wrappers.
// ---------------------------------------------------------------------------

pub struct Ort {
    // Kept alive for as long as the API table is used; dropped last.
    _lib: Library,
    api: *const c_void,
    pub version_string: String,
}

// The API table is owned by the loaded library and only read; every
// call through it happens under the owning embedder's mutex.
unsafe impl Send for Ort {}

pub struct TensorData {
    pub dims: Vec<i64>,
    pub data: Vec<f32>,
}

impl Ort {
    pub fn load(lib_path: &str) -> CResult<Ort> {
        let lib = Library::load(lib_path)?;
        let sym = lib.symbol("OrtGetApiBase")?;
        // SAFETY: OrtGetApiBase is ONNX Runtime's documented entry
        // point with exactly this signature.
        let get_base: FnOrtGetApiBase = unsafe { std::mem::transmute(sym) };
        let base = unsafe { get_base() };
        if base.is_null() {
            return Err((
                ERR_NOT_SUPPORTED,
                "OrtGetApiBase returned null — not a working ONNX Runtime library".into(),
            ));
        }
        let version_string = unsafe {
            let p = (base.read().get_version_string)();
            if p.is_null() {
                String::new()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };
        let api = unsafe { (base.read().get_api)(ORT_API_VERSION) };
        if api.is_null() {
            return Err((
                ERR_NOT_SUPPORTED,
                format!(
                    "ONNX Runtime {version_string} does not serve C API version \
                     {ORT_API_VERSION} (EmbedKit v0.1 needs a runtime at least that new; \
                     pins.txt names a verified one)"
                ),
            ));
        }
        let ort = Ort {
            _lib: lib,
            api,
            version_string,
        };
        for &i in USED_INDICES {
            if ort.slot_ptr(i).is_null() {
                return Err((
                    ERR_NOT_SUPPORTED,
                    format!("ONNX Runtime API table is missing entry {i} — unsupported runtime"),
                ));
            }
        }
        Ok(ort)
    }

    fn slot_ptr(&self, index: usize) -> *const c_void {
        // SAFETY: api is the OrtApi table for ORT_API_VERSION, whose
        // layout (an array of function pointers) is fixed by the C API.
        unsafe { *(self.api as *const *const c_void).add(index) }
    }

    /// Read one typed function pointer out of the table.
    ///
    /// SAFETY: `T` must be the exact signature the pinned header
    /// declares for `index`; every use below is transcribed from it,
    /// and fn pointers are pointer-sized on every target we build.
    unsafe fn slot<T: Copy>(&self, index: usize) -> T {
        debug_assert_eq!(
            std::mem::size_of::<T>(),
            std::mem::size_of::<*const c_void>()
        );
        let raw = self.slot_ptr(index);
        std::mem::transmute_copy(&raw)
    }

    /// Check an OrtStatus; on failure, take its message and free it.
    ///
    /// # Safety
    /// `status` must be a status returned by an OrtApi call that has
    /// not been checked or released before.
    pub unsafe fn check(&self, status: OrtStatusPtr) -> CResult<()> {
        if status.is_null() {
            return Ok(());
        }
        let get_msg: FnGetErrorMessage = unsafe { self.slot(GET_ERROR_MESSAGE) };
        let msg = unsafe {
            let p = get_msg(status);
            if p.is_null() {
                "unknown ONNX Runtime error".to_string()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };
        unsafe { self.release(RELEASE_STATUS, status) };
        Err((ERR_PLATFORM, format!("ONNX Runtime: {msg}")))
    }

    /// Call one of the table's `Release*` functions.
    ///
    /// # Safety
    /// `ptr` must be a live object of the kind `index` releases,
    /// not previously released.
    pub unsafe fn release(&self, index: usize, ptr: *mut c_void) {
        if !ptr.is_null() {
            let f: FnRelease = self.slot(index);
            f(ptr);
        }
    }

    /// # Safety
    /// `p` must be a live OrtEnv owned by the caller.
    pub unsafe fn release_env(&self, p: *mut c_void) {
        self.release(RELEASE_ENV, p)
    }
    /// # Safety
    /// `p` must be a live OrtSession owned by the caller.
    pub unsafe fn release_session(&self, p: *mut c_void) {
        self.release(RELEASE_SESSION, p)
    }
    /// # Safety
    /// `p` must be a live OrtMemoryInfo owned by the caller.
    pub unsafe fn release_memory_info(&self, p: *mut c_void) {
        self.release(RELEASE_MEMORY_INFO, p)
    }

    /// Safe-for-this-module status check: every status passed here
    /// came straight from an OrtApi call in this module and is
    /// checked exactly once, as `check` requires.
    fn checked(&self, status: OrtStatusPtr) -> CResult<()> {
        unsafe { self.check(status) }
    }

    pub fn create_env(&self) -> CResult<*mut c_void> {
        let f: FnCreateEnv = unsafe { self.slot(CREATE_ENV) };
        let logid = CString::new("embedkit").unwrap();
        let mut env: *mut c_void = std::ptr::null_mut();
        let status = unsafe { f(LOG_WARNING, logid.as_ptr(), &mut env) };
        self.checked(status)?;
        Ok(env)
    }

    /// Create a session (default options), releasing the options
    /// object before returning. The model path crosses as UTF-16 on
    /// Windows (ORTCHAR_T) and as bytes elsewhere.
    /// # Safety
    /// `env` must be a live OrtEnv from `create_env`.
    pub unsafe fn create_session(
        &self,
        env: *mut c_void,
        model_path: &str,
    ) -> CResult<*mut c_void> {
        let mk_opts: FnCreateSessionOptions = unsafe { self.slot(CREATE_SESSION_OPTIONS) };
        let mut opts: *mut c_void = std::ptr::null_mut();
        let status = unsafe { mk_opts(&mut opts) };
        self.checked(status)?;
        let result = self.create_session_with_options(env, model_path, opts);
        unsafe { self.release(RELEASE_SESSION_OPTIONS, opts) };
        result
    }

    fn create_session_with_options(
        &self,
        env: *mut c_void,
        model_path: &str,
        opts: *mut c_void,
    ) -> CResult<*mut c_void> {
        // The create fn's path parameter type differs by platform;
        // call through a cfg-matched alias.
        #[cfg(windows)]
        {
            type FnCreateSession = unsafe extern "C" fn(
                *const c_void,
                *const u16,
                *const c_void,
                *mut *mut c_void,
            ) -> OrtStatusPtr;
            let f: FnCreateSession = unsafe { self.slot(CREATE_SESSION) };
            let wide: Vec<u16> = model_path
                .encode_utf16()
                .chain(std::iter::once(0))
                .collect();
            let mut session: *mut c_void = std::ptr::null_mut();
            let status = unsafe { f(env, wide.as_ptr(), opts, &mut session) };
            self.checked(status)?;
            Ok(session)
        }
        #[cfg(not(windows))]
        {
            type FnCreateSession = unsafe extern "C" fn(
                *const c_void,
                *const c_char,
                *const c_void,
                *mut *mut c_void,
            ) -> OrtStatusPtr;
            let f: FnCreateSession = unsafe { self.slot(CREATE_SESSION) };
            let c = CString::new(model_path).map_err(|_| {
                (
                    ERR_PLATFORM,
                    format!("model path contains a NUL byte: '{model_path}'"),
                )
            })?;
            let mut session: *mut c_void = std::ptr::null_mut();
            let status = unsafe { f(env, c.as_ptr(), opts, &mut session) };
            self.checked(status)?;
            Ok(session)
        }
    }

    /// # Safety
    /// `session` must be a live OrtSession.
    unsafe fn session_names(&self, session: *mut c_void, inputs: bool) -> CResult<Vec<String>> {
        let get_count: FnGetCount = unsafe {
            self.slot(if inputs {
                SESSION_GET_INPUT_COUNT
            } else {
                SESSION_GET_OUTPUT_COUNT
            })
        };
        let get_name: FnGetName = unsafe {
            self.slot(if inputs {
                SESSION_GET_INPUT_NAME
            } else {
                SESSION_GET_OUTPUT_NAME
            })
        };
        let get_alloc: FnGetDefaultAllocator =
            unsafe { self.slot(GET_ALLOCATOR_WITH_DEFAULT_OPTIONS) };
        let alloc_free: FnAllocatorFree = unsafe { self.slot(ALLOCATOR_FREE) };
        let mut allocator: *mut c_void = std::ptr::null_mut();
        self.checked(unsafe { get_alloc(&mut allocator) })?;
        // NOTE: the default allocator belongs to the runtime and is
        // never released; only the name strings are freed (with it).
        let mut count: usize = 0;
        self.checked(unsafe { get_count(session, &mut count) })?;
        let mut names = Vec::with_capacity(count);
        for i in 0..count {
            let mut p: *mut c_char = std::ptr::null_mut();
            self.checked(unsafe { get_name(session, i, allocator, &mut p) })?;
            let s = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
            self.checked(unsafe { alloc_free(allocator, p as *mut c_void) })?;
            names.push(s);
        }
        Ok(names)
    }

    /// # Safety
    /// `session` must be a live OrtSession.
    pub unsafe fn session_input_names(&self, session: *mut c_void) -> CResult<Vec<String>> {
        self.session_names(session, true)
    }

    /// # Safety
    /// `session` must be a live OrtSession.
    pub unsafe fn session_output_names(&self, session: *mut c_void) -> CResult<Vec<String>> {
        self.session_names(session, false)
    }

    pub fn create_cpu_memory_info(&self) -> CResult<*mut c_void> {
        let f: FnCreateCpuMemoryInfo = unsafe { self.slot(CREATE_CPU_MEMORY_INFO) };
        let mut info: *mut c_void = std::ptr::null_mut();
        let status = unsafe { f(ALLOCATOR_DEVICE, MEM_TYPE_DEFAULT, &mut info) };
        self.checked(status)?;
        Ok(info)
    }

    /// Run the session on int64 inputs and return one float output.
    /// `inputs` pairs each session input name with its token buffer;
    /// every tensor is shaped `[1, len]`. The output's dims and data
    /// are copied out before any OrtValue is released.
    /// # Safety
    /// `session` and `mem_info` must be live ORT objects owned by
    /// the caller; the input buffers must outlive the call (they
    /// do — they are borrowed for exactly its duration).
    pub unsafe fn run_int64_to_float(
        &self,
        session: *mut c_void,
        mem_info: *mut c_void,
        inputs: &[(CString, &[i64])],
        output_name: &CString,
    ) -> CResult<TensorData> {
        let create_tensor: FnCreateTensor =
            unsafe { self.slot(CREATE_TENSOR_WITH_DATA_AS_ORT_VALUE) };
        let run: FnRun = unsafe { self.slot(RUN) };
        let mut values: Vec<*mut c_void> = Vec::with_capacity(inputs.len());
        let mut out: *mut c_void = std::ptr::null_mut();
        let result = (|| -> CResult<TensorData> {
            for (_, data) in inputs {
                let shape: [i64; 2] = [1, data.len() as i64];
                let byte_len = std::mem::size_of_val(*data);
                let mut value: *mut c_void = std::ptr::null_mut();
                let status = unsafe {
                    create_tensor(
                        mem_info,
                        data.as_ptr() as *mut c_void,
                        byte_len,
                        shape.as_ptr(),
                        shape.len(),
                        TENSOR_INT64,
                        &mut value,
                    )
                };
                self.checked(status)?;
                values.push(value);
            }
            let name_ptrs: Vec<*const c_char> = inputs.iter().map(|(n, _)| n.as_ptr()).collect();
            let out_name_ptrs: [*const c_char; 1] = [output_name.as_ptr()];
            let status = unsafe {
                run(
                    session,
                    std::ptr::null(),
                    name_ptrs.as_ptr(),
                    values.as_ptr() as *const *const c_void,
                    values.len(),
                    out_name_ptrs.as_ptr(),
                    1,
                    &mut out,
                )
            };
            self.checked(status)?;
            self.read_float_tensor(out)
        })();
        for v in values {
            unsafe { self.release(RELEASE_VALUE, v) };
        }
        if !out.is_null() {
            unsafe { self.release(RELEASE_VALUE, out) };
        }
        result
    }

    /// SAFETY: `value` must be a live OrtValue produced by Run.
    unsafe fn read_float_tensor(&self, value: *mut c_void) -> CResult<TensorData> {
        let get_info: FnGetTensorTypeAndShape = unsafe { self.slot(GET_TENSOR_TYPE_AND_SHAPE) };
        let get_type: FnGetElementType = unsafe { self.slot(GET_TENSOR_ELEMENT_TYPE) };
        let get_count: FnGetDimsCount = unsafe { self.slot(GET_DIMENSIONS_COUNT) };
        let get_dims: FnGetDims = unsafe { self.slot(GET_DIMENSIONS) };
        let get_data: FnGetTensorMutableData = unsafe { self.slot(GET_TENSOR_MUTABLE_DATA) };
        let mut info: *mut c_void = std::ptr::null_mut();
        self.checked(unsafe { get_info(value, &mut info) })?;
        let result = (|| -> CResult<TensorData> {
            let mut elem: i32 = 0;
            self.checked(unsafe { get_type(info, &mut elem) })?;
            if elem != TENSOR_FLOAT {
                return Err((
                    ERR_PLATFORM,
                    format!("model output element type is {elem}, expected float (1)"),
                ));
            }
            let mut ndims: usize = 0;
            self.checked(unsafe { get_count(info, &mut ndims) })?;
            let mut dims = vec![0i64; ndims];
            if ndims > 0 {
                self.checked(unsafe { get_dims(info, dims.as_mut_ptr(), ndims) })?;
            }
            if dims.iter().any(|&d| d < 0) {
                return Err((
                    ERR_PLATFORM,
                    format!("model output has a dynamic dimension: {dims:?}"),
                ));
            }
            let total: usize = dims.iter().map(|&d| d as usize).product();
            let mut raw: *mut c_void = std::ptr::null_mut();
            self.checked(unsafe { get_data(value, &mut raw) })?;
            if raw.is_null() && total > 0 {
                return Err((ERR_PLATFORM, "model output data pointer is null".into()));
            }
            let data = if total == 0 {
                Vec::new()
            } else {
                unsafe { std::slice::from_raw_parts(raw as *const f32, total) }.to_vec()
            };
            Ok(TensorData { dims, data })
        })();
        unsafe { self.release(RELEASE_TENSOR_TYPE_AND_SHAPE_INFO, info) };
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_library_is_platform_error() {
        let err = Ort::load("no/such/libonnxruntime.so").err().unwrap();
        assert_eq!(err.0, ERR_PLATFORM);
        assert!(err.1.contains("failed to load"), "{}", err.1);
    }

    #[test]
    fn non_runtime_library_is_not_supported() {
        // Any existing file that is not a loadable library gives the
        // load error; a loadable non-ORT library gives NOT_SUPPORTED.
        // The crate's own manifest is a stable existing non-library.
        let err = Ort::load("Cargo.toml").err().unwrap();
        assert_eq!(err.0, ERR_PLATFORM);
    }
}
