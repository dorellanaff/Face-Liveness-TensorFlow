import uvicorn

if __name__ == "__main__":
    uvicorn.run('main:app', host="0.0.0.0", port=8081, reload=True,
        #ssl_keyfile='certs/localhost+2-key.pem',
        #ssl_certfile='certs/localhost+2.pem'
    )
