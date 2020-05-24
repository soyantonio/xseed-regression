#!/bin/sh

if [ ! -f "env/.env" ]; then
    printf "🔨 Creating env file\n"
    cp env/.env.example env/.env
else
    printf "✔️ Env file already exists\n"
fi


if [ ! -d "img" ]; then
    printf "🔨 Creating img folder\n"
    mkdir img
else
    printf "✔️ img folder already exists\n"
fi

if [ ! -d "data" ]; then
    printf "🔨 Creating data folder\n"
    mkdir img
else
    printf "✔️ data folder already exists\n"
fi




printf "Done 😎\n"