# Building Docker

```
docker build -t wordshk_server .
```

# Upload to Hub
```
docker tag wordshk_server:latest alienkevin/wordshk-server:latest
docker push alienkevin/wordshk-server:latest
```

# Running Docker

```
docker run -p 3000:3000 wordshk_server
```

# Setup SSH to AWS Beanstalk
1. Install EB CLI
```
brew install awsebcli
```
2. Init
```
eb init
```
3. Setup SSH
```
eb ssh --setup
```
4. Connect through SSH
```
eb ssh
```

# Issue Dev SSL certificate

```
openssl genrsa -out wordshk_server.key 2048
openssl req -new -key wordshk_server.key -out wordshk_server.csr
openssl x509 -req -days 365 -in wordshk_server.csr -signkey wordshk_server.key -out wordshk_server.crt
aws iam upload-server-certificate --server-certificate-name wordshk_server_cert --certificate-body file://wordshk_server.crt --private-key file://wordshk_server.key
```
