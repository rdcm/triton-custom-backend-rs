use crate::error::Error;
use crate::model::Model;
use crate::request::Request;

pub trait Backend {
    /// Initialize a backend. This function is optional, a backend is not
    /// required to implement it. This function is called once when a
    /// backend is loaded to allow the backend to initialize any state
    /// associated with the backend. A backend has a single state that is
    /// shared across all models that use the backend.
    ///
    /// Corresponds to TRITONBACKEND_Initialize.
    fn initialize() -> Result<(), Error> {
        Ok(())
    }

    /// Finalize for a backend. This function is optional, a backend is
    /// not required to implement it. This function is called once, just
    /// before the backend is unloaded. All state associated with the
    /// backend should be freed and any threads created for the backend
    /// should be exited/joined before returning from this function.
    /// Corresponds to TRITONBACKEND_Finalize.
    fn finalize() -> Result<(), Error> {
        Ok(())
    }

    /// Initialize for a model instance. This function is optional, a
    /// backend is not required to implement it. This function is called
    /// once when a model instance is created to allow the backend to
    /// initialize any state associated with the instance.
    ///
    /// Corresponds to TRITONBACKEND_ModelInstanceInitialize.
    fn model_instance_initialize() -> Result<(), Error> {
        Ok(())
    }

    /// Finalize for a model instance. This function is optional, a
    /// backend is not required to implement it. This function is called
    /// once for an instance, just before the corresponding model is
    /// unloaded from Triton. All state associated with the instance
    /// should be freed and any threads created for the instance should be
    /// exited/joined before returning from this function.
    ///
    /// Corresponds to TRITONBACKEND_ModelInstanceFinalize.
    fn model_instance_finalize() -> Result<(), Error> {
        Ok(())
    }

    /// Execute a batch of one or more requests on a model instance. This
    /// function is required. Triton will not perform multiple
    /// simultaneous calls to this function for a given model 'instance';
    /// however, there may be simultaneous calls for different model
    /// instances (for the same or different models).
    ///
    /// Corresponds to TRITONBACKEND_ModelInstanceExecute.
    fn model_instance_execute(model: Model, requests: &[Request]) -> Result<(), Error>;
}

#[macro_export]
macro_rules! call_checked {
    ($res:expr) => {
        match $res {
            Err(err) => {
                let err = std::ffi::CString::new(err.to_string()).expect("CString::new failed");
                unsafe {
                    triton_ng::sys::TRITONSERVER_ErrorNew(
                        triton_ng::sys::TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_INTERNAL,
                        err.as_ptr(),
                    )
                }
            }
            Ok(ok) => std::ptr::null(),
        }
    };
}

#[macro_export]
macro_rules! declare_backend {
    ($class:ident) => {
        #[unsafe(no_mangle)]
        extern "C" fn TRITONBACKEND_Initialize(
            backend: *const triton_ng::sys::TRITONBACKEND_Backend,
        ) -> *const triton_ng::sys::TRITONSERVER_Error {
            triton_ng::call_checked!($class::initialize())
        }

        #[unsafe(no_mangle)]
        extern "C" fn TRITONBACKEND_Finalize(
            backend: *const triton_ng::sys::TRITONBACKEND_Backend,
        ) -> *const triton_ng::sys::TRITONSERVER_Error {
            triton_ng::call_checked!($class::finalize())
        }

        #[unsafe(no_mangle)]
        extern "C" fn TRITONBACKEND_ModelInstanceInitialize(
            instance: *mut triton_ng::sys::TRITONBACKEND_ModelInstance,
        ) -> *const triton_ng::sys::TRITONSERVER_Error {
            triton_ng::call_checked!($class::model_instance_initialize())
        }

        #[unsafe(no_mangle)]
        extern "C" fn TRITONBACKEND_ModelInstanceFinalize(
            instance: *const triton_ng::sys::TRITONBACKEND_ModelInstance,
        ) -> *const triton_ng::sys::TRITONSERVER_Error {
            triton_ng::call_checked!($class::model_instance_finalize())
        }

        #[unsafe(no_mangle)]
        extern "C" fn TRITONBACKEND_ModelInstanceExecute(
            instance: *mut triton_ng::sys::TRITONBACKEND_ModelInstance,
            requests: *const *mut triton_ng::sys::TRITONBACKEND_Request,
            request_count: u32,
        ) -> *const triton_ng::sys::TRITONSERVER_Error {
            let mut model: *mut triton_ng::sys::TRITONBACKEND_Model = std::ptr::null_mut();
            let err =
                unsafe { triton_ng::sys::TRITONBACKEND_ModelInstanceModel(instance, &mut model) };
            if !err.is_null() {
                return err;
            }

            let model = triton_ng::Model::from_ptr(model);

            let requests = unsafe { std::slice::from_raw_parts(requests, request_count as usize) };
            let requests = requests
                .iter()
                .map(|req| triton_ng::Request::from_ptr(*req))
                .collect::<Vec<triton_ng::Request>>();

            triton_ng::call_checked!($class::model_instance_execute(model, &requests))
        }
    };
}
