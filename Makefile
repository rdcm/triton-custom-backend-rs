docker-env-up:
	docker compose up -d

docker-env-down:
	docker compose down

format:
	cargo fmt

build:
	cargo build --release
	mv target/release/libtriton_custom_backend.so backends/custom_backend/libtriton_custom_backend.so

logs:
	docker logs triton

update-submodules:
	git submodule update --remote


run:
	# https://github.com/triton-inference-server/common/tree/main/protobuf
	cargo build --manifest-path=triton-client/Cargo.toml --release
	cargo build --workspace
	cargo run ./target/debug/app