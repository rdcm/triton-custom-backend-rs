use crate::error::{Error, TritonError};
use crate::ffi_call;
use crate::server::Server;
use crate::utils::cstr_to_string;
use libc::c_char;
use std::fs::File;
use std::io::prelude::*;
use std::path::PathBuf;
use std::ptr;

pub struct Model {
    ptr: *mut triton_sys::TRITONBACKEND_Model,
}

impl Model {
    pub fn from_ptr(ptr: *mut triton_sys::TRITONBACKEND_Model) -> Self {
        Self { ptr }
    }

    pub fn name(&self) -> Result<String, TritonError> {
        let mut model_name: *const c_char = std::ptr::null_mut();
        ffi_call!(triton_sys::TRITONBACKEND_ModelName(
            self.ptr,
            &mut model_name
        ))?;

        Ok(unsafe { cstr_to_string(model_name) })
    }

    pub fn version(&self) -> Result<u64, TritonError> {
        let mut version = 0u64;
        ffi_call!(triton_sys::TRITONBACKEND_ModelVersion(
            self.ptr,
            &mut version
        ))?;

        Ok(version)
    }

    pub fn location(&self) -> Result<String, TritonError> {
        let mut artifact_type: triton_sys::TRITONBACKEND_ArtifactType = 0u32;
        let mut location: *const c_char = std::ptr::null_mut();
        ffi_call!(triton_sys::TRITONBACKEND_ModelRepository(
            self.ptr,
            &mut artifact_type,
            &mut location
        ))?;

        Ok(unsafe { cstr_to_string(location) })
    }

    pub fn path(&self, filename: &str) -> Result<PathBuf, Error> {
        Ok(PathBuf::from(format!(
            "{}/{}/{}",
            self.location()?,
            self.version()?,
            filename
        )))
    }

    pub fn load_file(&self, filename: &str) -> Result<Vec<u8>, Error> {
        let path = self.path(filename)?;
        let mut f = File::open(path)?;

        let mut buffer = Vec::new();
        f.read_to_end(&mut buffer)?;

        Ok(buffer)
    }

    pub fn get_server(&self) -> Result<Server, TritonError> {
        let mut server_ptr: *mut triton_sys::TRITONSERVER_Server = ptr::null_mut();

        ffi_call!(
            triton_sys::TRITONBACKEND_ModelServer(self.ptr, &mut server_ptr),
            server_ptr
        )?;

        Server::from_ptr(server_ptr)
    }

    pub fn as_ptr(&self) -> *mut triton_sys::TRITONBACKEND_Model {
        self.ptr
    }
}
