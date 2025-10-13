docker-env-up:
	docker compose up -d

docker-env-down:
	docker compose down

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

build-debug:
	cargo build --workspace