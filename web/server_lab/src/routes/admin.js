const express = require("express");
const adminRouter = express.Router();
const adminService = require("../services/adminService");
adminRouter.post("/register", adminService.createAdmin);
adminRouter.post("/login", adminService.loginAdmin);
module.exports = adminRouter;
