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
