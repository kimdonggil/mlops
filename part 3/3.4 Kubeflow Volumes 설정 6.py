apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: example-claim
spec:
  storageClassName: manual
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 95Gi