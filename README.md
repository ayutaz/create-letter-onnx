# create-letter-onnx

# how to use

## set up docker
1. install docker for your OS
2. `docker compose up -d --build`
3. `docker compose exec python3 bash`

## finish docker
1. `exit` in terminal(It is assumed that you are within the dockr environment
2. `docker compose down`

## delete docker
1. `docker image ls`
2. `docker image rm imageid`

# Reference Site
* [DockerでPython実行環境を作ってみる](https://qiita.com/jhorikawa_err/items/fb9c03c0982c29c5b6d5#step-8-%E3%82%B3%E3%83%B3%E3%83%86%E3%83%8A%E3%81%AE%E5%89%8A%E9%99%A4)