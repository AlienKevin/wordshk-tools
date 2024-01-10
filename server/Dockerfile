# Start from the official Rust image
FROM rust:latest

# Create a new empty shell project
RUN USER=root cargo new --bin wordshk_server
WORKDIR /wordshk_server

# Copy your source code into the Docker image
COPY ./ ./

# Build your application
RUN cargo build --release

# The command to run your application
CMD ["./target/release/wordshk_server"]
