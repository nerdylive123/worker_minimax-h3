// docker buildx bake targets for the MiniMax-H3 worker.
// All values are overridable from CI via `set: *.args.*`.

variable "DOCKERHUB_REPO" {
  default = "nerdylive123"
}

variable "DOCKERHUB_IMG" {
  default = "worker-minimax-h3"
}

variable "RELEASE_VERSION" {
  default = "latest"
}

variable "HUGGINGFACE_ACCESS_TOKEN" {
  default = ""
}

target "worker-minimax-h3" {
  dockerfile = "Dockerfile"
  context    = "."
  platforms  = ["linux/amd64"]
  tags       = ["${DOCKERHUB_REPO}/${DOCKERHUB_IMG}:${RELEASE_VERSION}"]
  secret = [
    "id=HF_TOKEN,env=HUGGINGFACE_ACCESS_TOKEN"
  ]
}
