docker-env-up:
	mkdir ./backends/custom_backend -p
	docker compose up -d

docker-env-down:
	docker compose down

docker-env-clean:
	docker compose down -v

format:
	cargo fmt

lint:
	cargo clippy

logs:
	docker logs triton

update-submodules:
	git submodule update --remote

gen-grpc-client:
	cargo build --manifest-path=triton-grpc-client/Cargo.toml --release

build:
	cargo build --release
	mv target/release/libtriton_custom_backend.so backends/custom_backend/libtriton_custom_backend.so

download-model:
	mkdir models/mnist_onnx/1/ -p
	wget https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx
	mv mnist-12.onnx models/mnist_onnx/1/model.onnx
