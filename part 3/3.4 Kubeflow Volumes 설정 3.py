apiVersion: v1
kind: PersistentVolume
metadata:
  name: example-volume
spec:
  storageClassName: manual
  capacity:
    storage: 100Gi
  volumeMode: Filesystem
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Delete
  hostPath:
    path: /mnt/working/kubeflow/volumes