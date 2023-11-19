const express = require("express");
const middleware = require("./middleware");
const userRouter = express.Router();
const userService = require("../services/userService");
userRouter.post("/register", middleware.middleware, userService.createUser);
userRouter.get("/full", middleware.middleware, userService.getFullUser);
module.exports = userRouter;
