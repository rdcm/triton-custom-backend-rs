use anyhow::Result;
use tonic::Request;
use tonic::transport::Channel;
use triton_grpc_client::inference::grpc_inference_service_client::GrpcInferenceServiceClient;
use triton_grpc_client::inference::{ModelReadyRequest, ServerReadyRequest};

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
}
