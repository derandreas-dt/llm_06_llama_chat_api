Chat system which mimics the chat.openai.com services

Contains
    * an api server that handles the api calls with llm backend
    * an http static service to serve the html / css / js
    * a simple website to have a ui

This is terriable code at the frontend side, but wanted to have
a very simple pure js version, without some nasty frontend framework.
With something like vue/react this would be much more easy, but
didn't want that pipeline and it should be working with fastapi directly.

The frontend allow some features like multiple chats.
All chats are stored only on the client side (localstorage) and
by deleting it, its all gone.

Also on the backend side there is only a memory store. Restarting
the app prunes that memory and the messages in the client are not
sync with the backend anymore.

