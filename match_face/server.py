import uvicorn

if __name__ == "__main__":
    uvicorn.run('main:app', host="0.0.0.0", port=8081, reload=True,
<<<<<<< HEAD
        #ssl_keyfile='certs/localhost+2-key.pem',
        #ssl_certfile='certs/localhost+2.pem'
    )
=======
        ssl_keyfile='certs/localhost+2-key.pem',
        ssl_certfile='certs/localhost+2.pem')
>>>>>>> e390cfcf0533049c9814e6f38809ac5c19490748
