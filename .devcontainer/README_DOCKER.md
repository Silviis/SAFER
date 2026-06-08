Docker build notes for the Drone Surveillance devcontainer

- **Build**: from workspace root run:

```
docker build -f .devcontainer/Dockerfile -t safer_ws:humble --build-arg WORKSPACE=/workspaces/safer_ws .
```

- **Optional Olympe**: if you have a Parrot Olympe .deb URL, pass it as a build-arg:

```
docker build -f .devcontainer/Dockerfile -t safer_ws:humble \
  --build-arg WORKSPACE=/workspaces/safer_ws \
  --build-arg OLYMPE_DEB_URL="https://example.com/olympe.deb" .
```

- **Devcontainer**: VS Code remote containers will use the Dockerfile automatically if configured.

- **Notes**:
  - The image installs CPU PyTorch wheels by default. Adjust the Dockerfile if you need CUDA-enabled wheels.
  - Olympe is not distributed via PyPI — supply a Parrot .deb package URL to the build if you want it preinstalled.
  - `cv_bridge` is installed from `apt` when available; Python OpenCV is provided via `opencv-python-headless`.
