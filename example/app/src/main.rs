mod request;

use crate::request::ModelTensors;
use anyhow::Result;
use triton_client::{InferenceOutput, TritonClient};

#[tokio::main]
async fn main() -> Result<()> {
    let mut client = TritonClient::new("http://localhost:8001").await?;
    let server_is_ready = client.server_ready().await?;
    println!("Server is ready: {}", server_is_ready);

    let model_is_ready = client.model_ready("mnist", "1").await?;
    println!("Model is ready: {}", model_is_ready);

    let response = client
        .infer(ModelTensors::new("mnist", vec![0.123f32; 784]).into())
        .await?;

    let output = InferenceOutput::from_response(&response)?;
    println!("Output: {:?}", output);

    Ok(())
}
