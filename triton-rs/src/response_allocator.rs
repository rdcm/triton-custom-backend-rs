use crate::TritonError;
use std::alloc::{Layout, alloc, dealloc};
use std::os::raw::{c_char, c_void};
use std::ptr;

pub struct ResponseAllocator {
    ptr: *mut triton_sys::TRITONSERVER_ResponseAllocator,
}

impl ResponseAllocator {
    pub fn new() -> Result<Self, TritonError> {
        let mut allocator_ptr: *mut triton_sys::TRITONSERVER_ResponseAllocator = ptr::null_mut();

        ffi_call!(triton_sys::TRITONSERVER_ResponseAllocatorNew(
            &mut allocator_ptr,
            Some(response_allocator_fn),
            Some(response_release_fn),
            None,
        ))?;

        ensure_ptr!(allocator_ptr)?;

        Ok(Self { ptr: allocator_ptr })
    }

    pub fn as_ptr(&self) -> *mut triton_sys::TRITONSERVER_ResponseAllocator {
        self.ptr
    }
}

unsafe extern "C" fn response_allocator_fn(
    _allocator: *mut triton_sys::TRITONSERVER_ResponseAllocator,
    _tensor_name: *const c_char,
    byte_size: usize,
    _memory_type: triton_sys::TRITONSERVER_MemoryType,
    _memory_type_id: i64,
    _userp: *mut c_void,
    buffer: *mut *mut c_void,
    buffer_userp: *mut *mut c_void,
    actual_memory_type: *mut triton_sys::TRITONSERVER_MemoryType,
    actual_memory_type_id: *mut i64,
) -> *mut triton_sys::TRITONSERVER_Error {
    let layout = match Layout::from_size_align(byte_size, 8) {
        Ok(l) => l,
        Err(_) => {
            return unsafe {
                triton_sys::TRITONSERVER_ErrorNew(
                    triton_sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_INTERNAL,
                    c"Invalid layout".as_ptr() as *const c_char,
                )
            };
        }
    };

    let buf = unsafe { alloc(layout) as *mut c_void };

    if buf.is_null() {
        return unsafe {
            triton_sys::TRITONSERVER_ErrorNew(
                triton_sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_INTERNAL,
                c"Failed to allocate memory".as_ptr() as *const c_char,
            )
        };
    }

    // TODO: TRITONSERVER_MEMORY_GPU
    unsafe {
        *buffer = buf;
        *buffer_userp = ptr::null_mut();
        *actual_memory_type = triton_sys::TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU;
        *actual_memory_type_id = 0;
    }

    ptr::null_mut()
}

unsafe extern "C" fn response_release_fn(
    _allocator: *mut triton_sys::TRITONSERVER_ResponseAllocator,
    buffer: *mut c_void,
    _buffer_userp: *mut c_void,
    byte_size: usize,
    _memory_type: triton_sys::TRITONSERVER_MemoryType,
    _memory_type_id: i64,
) -> *mut triton_sys::TRITONSERVER_Error {
    if !buffer.is_null() && byte_size > 0 {
        if let Ok(layout) = Layout::from_size_align(byte_size, 8) {
            unsafe { dealloc(buffer as *mut u8, layout) };
        }
    }
    ptr::null_mut()
}
