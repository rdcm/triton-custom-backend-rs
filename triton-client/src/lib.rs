use anyhow::Result;
use tonic::Request;
use tonic::transport::Channel;
use triton_grpc_client::inference::grpc_inference_service_client::GrpcInferenceServiceClient;
use triton_grpc_client::inference::{
    ModelInferRequest, ModelInferResponse, ModelReadyRequest, ServerReadyRequest,
};

#[derive(Debug)]
pub struct InferenceOutput {
    pub name: String,
    pub datatype: String,
    pub shape: Vec<i64>,
    pub data: OutputData,
}

#[derive(Debug)]
pub enum OutputData {
    FP32(Vec<f32>),
    FP64(Vec<f64>),
    INT32(Vec<i32>),
    INT64(Vec<i64>),
    Bytes(Vec<String>),
    Unknown(Vec<u8>),
}

impl InferenceOutput {
    pub fn from_response(response: &ModelInferResponse) -> Result<Vec<Self>> {
        let mut outputs = Vec::new();

        for (i, output_meta) in response.outputs.iter().enumerate() {
            let raw_data = response
                .raw_output_contents
                .get(i)
                .ok_or_else(|| anyhow::anyhow!("Missing raw data for output {}", i))?;

            let data = match output_meta.datatype.as_str() {
                "FP32" => OutputData::FP32(decode_fp32_tensor(raw_data)),
                "FP64" => OutputData::FP64(decode_fp64_tensor(raw_data)),
                "INT32" => OutputData::INT32(decode_int32_tensor(raw_data)),
                "INT64" => OutputData::INT64(decode_int64_tensor(raw_data)),
                "BYTES" => OutputData::Bytes(decode_string_tensor(raw_data)),
                _ => OutputData::Unknown(raw_data.to_vec()),
            };

            outputs.push(InferenceOutput {
                name: output_meta.name.clone(),
                datatype: output_meta.datatype.clone(),
                shape: output_meta.shape.clone(),
                data,
            });
        }

        Ok(outputs)
    }
}

fn decode_string_tensor(_data: &[u8]) -> Vec<String> {
    todo!()
}

fn decode_int64_tensor(_data: &[u8]) -> Vec<i64> {
    todo!()
}

fn decode_int32_tensor(_data: &[u8]) -> Vec<i32> {
    todo!()
}

fn decode_fp64_tensor(_data: &[u8]) -> Vec<f64> {
    todo!()
}

fn decode_fp32_tensor(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| {
            let bytes: [u8; 4] = chunk.try_into().unwrap();
            f32::from_le_bytes(bytes)
        })
        .collect()
}

pub struct TritonClient {
    client: GrpcInferenceServiceClient<Channel>,
}

impl TritonClient {
    pub async fn new(url: &str) -> Result<Self> {
        let channel = Channel::from_shared(url.to_string())?.connect().await?;
        let client = GrpcInferenceServiceClient::new(channel);

        Ok(TritonClient { client })
    }

    pub async fn server_ready(&mut self) -> Result<bool> {
        let request = Request::new(ServerReadyRequest {});
        let response = self.client.server_ready(request).await?;
        Ok(response.into_inner().ready)
    }

    pub async fn model_ready(&mut self, model_name: &str, model_version: &str) -> Result<bool> {
        let request = Request::new(ModelReadyRequest {
            name: model_name.to_string(),
            version: model_version.to_string(),
        });
        let response = self.client.model_ready(request).await?;
        Ok(response.into_inner().ready)
    }

    pub async fn infer(&mut self, request: ModelInferRequest) -> Result<ModelInferResponse> {
        let request = Request::new(request);
        let response = self.client.model_infer(request).await?;
        Ok(response.into_inner())
    }
}
