use triton_grpc_client::inference::model_infer_request::{
    InferInputTensor, InferRequestedOutputTensor,
};
use triton_grpc_client::inference::{InferTensorContents, ModelInferRequest};

pub struct ModelTensors {
    model_name: String,
    input_data: Vec<f32>,
}

impl ModelTensors {
    pub fn new(model_name: &str, input_data: Vec<f32>) -> Self {
        ModelTensors {
            model_name: model_name.to_string(),
            input_data,
        }
    }
}

impl From<ModelTensors> for ModelInferRequest {
    fn from(value: ModelTensors) -> Self {
        ModelInferRequest {
            model_name: value.model_name.to_string(),
            model_version: "".to_string(), // latest version
            id: "".to_string(),

            inputs: vec![InferInputTensor {
                name: "Input3".to_string(),
                datatype: "FP32".to_string(),
                shape: vec![1, 1, 28, 28],
                parameters: Default::default(),
                contents: Some(InferTensorContents {
                    fp32_contents: value.input_data,
                    ..Default::default()
                }),
            }],

            outputs: vec![InferRequestedOutputTensor {
                name: "Plus214_Output_0".to_string(),
                parameters: Default::default(),
            }],

            parameters: Default::default(),
            raw_input_contents: vec![],
        }
    }
}
