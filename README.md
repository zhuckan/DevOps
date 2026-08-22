# DevOps: CI/CD for ML Service

This project automates the build, test, and deployment of an ML service (from Task 1) using GitLab CE as the CI/CD tool and Minikube as the Kubernetes cluster.

## What is done

- **Minikube** with `ingress` and `metrics-server` addons enabled
- **Local GitLab CE** running in Docker, with a registered GitLab Runner
- **CI pipeline** in `.gitlab-ci.yml`:
  - runs tests with `pytest`
  - builds a Docker image with the FastAPI app
  - pushes the image to the GitLab Container Registry
- **Kubernetes setup**:
  - dedicated Namespace, ServiceAccount, Role, and RoleBinding (least privilege)
  - ResourceQuota and LimitRange for CPU/memory limits
  - Deployment, Service, and Ingress manifests
  - Liveness and Readiness probes for the model service
- **CD step** in `.gitlab-ci.yml`:
  - GitLab Runner authenticates to the cluster using Kubeconfig
  - applies Kubernetes manifests via `kubectl apply`

## Tech Stack

`Minikube (Kubernetes)` · `GitLab CE` · `GitLab Runner` · `Docker` · `Bash` · `kubectl`

## Repository Structure

- `.gitlab-ci.yml` — CI/CD pipeline definition
- `k8s/` — Kubernetes manifests (Namespace, RBAC, Quotas, Deployment, Service, Ingress)
- `Dockerfile` — Docker image for the ML service
- `app/` — FastAPI application code
- `tests/` — pytest tests

## How It Works

1. GitLab Runner runs tests and builds the Docker image.
2. The image is pushed to GitLab Container Registry.
3. The CD stage applies Kubernetes manifests to Minikube.
4. The service is exposed via Ingress and is ready to serve requests.
