#!/bin/sh
set -e

kubectl --kubeconfig "$KUBECONFIG" apply -f manifests/namespace.yaml
kubectl --kubeconfig "$KUBECONFIG" apply -f manifests/rbac.yaml
kubectl --kubeconfig "$KUBECONFIG" apply -f manifests/quota.yaml
kubectl --kubeconfig "$KUBECONFIG" apply -f manifests/limitrange.yaml
kubectl --kubeconfig "$KUBECONFIG" apply -f manifests/service.yaml

kubectl --kubeconfig "$KUBECONFIG" create secret docker-registry gitlab-registry \
  --docker-server=$CI_REGISTRY \
  --docker-username=$CI_REGISTRY_USER \
  --docker-password=$CI_REGISTRY_PASSWORD \
  --docker-email=ci@example.com \
  -n $KUBE_NAMESPACE \
  --dry-run=client -o yaml | kubectl --kubeconfig "$KUBECONFIG" apply -f -

sed "s|__CI_REGISTRY_IMAGE__|$CI_REGISTRY_IMAGE|g; s|__CI_COMMIT_SHORT_SHA__|$CI_COMMIT_SHORT_SHA|g" manifests/deployment.yaml | kubectl --kubeconfig "$KUBECONFIG" apply -f -

kubectl --kubeconfig "$KUBECONFIG" apply -f manifests/ingress.yaml