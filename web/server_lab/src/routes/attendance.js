const express = require("express");
const attendanceService = require("../services/attendanceService")
const attendanceRouter = express.Router();
attendanceRouter.post(
  "/create",
  attendanceService.createAttendance
);
attendanceRouter.post("/get", attendanceService.getFullAttendance);




module.exports = attendanceRouter;
