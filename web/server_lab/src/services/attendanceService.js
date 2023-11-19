const { Attendances } = require("../models");
const { Users } = require("../models");
const client = require("../mqtt");
const moment = require("moment");

client.client.subscribe("time_keeping_res");
client.client.subscribe("timekeep/time_keeping");
// client.client.subscribe("time_keeping_res");
client.client.on("message", async (topic, message) => {
  const now = moment();
  const formattedDate = now.format("DD/MM/YYYY");
  const gioPhut = now.format("HH:mm");
  const gio = now.format("HH"); // Định dạng 24 giờ
  const phut = now.format("mm");
  if (topic === "timekeep/time_keeping") {
    const response = message.toString();
    // resolve(response);
    console.log(response);
    const myArray = response.split("-");
    console.log(myArray[1]);
    const userCode = myArray[1];
    const checkUser = await Users.findOne({
      where: { userCode: userCode },
    });
    if (!checkUser) {
      client.client.publish("timekeep/time_keeping_res_host", "Error");
      return;
    } else {
      await Attendances.create({
        Hourmin: gioPhut,
        Time: formattedDate,
        UserId: checkUser.id,
      });
      client.client.publish(
        "timekeep/time_keeping_res_host",
        `${checkUser.userCode}`
      );
    }
  }
});

const createAttendance = async (req, res) => {
  const now = moment();
  const formattedDate = now.format("DD/MM/YYYY");
  const gioPhut = now.format("HH:mm");
  // const gio = now.format("HH"); // Định dạng 24 giờ
  // const phut = now.format("mm");
  // console.log(gioPhut);
  // console.log(gio);
  // console.log(phut);
  const { userCode } = req.body;
  if (!userCode) {
    return res.json({
      errCode: 1,
      message: "Dont enought information",
    });
  }
  try {
    const checkUser = await Users.findOne({
      where: { userCode: userCode },
    });
    if (!checkUser) {
      return res.json({
        errCode: 1,
        message: "Error,Check the Code again",
      });
    } else {
      // Tạo một promise để đợi phản hồi
      const waitForResponse = new Promise((resolve, reject) => {
        client.client.publish("time_keeping", `${userCode}`);
        setTimeout(() => {
          reject(new Error("Timeout waiting for response"));
        }, 2000); // Timeout sau 3 giây

        client.client.on("message", (topic, message) => {
          if (topic === "time_keeping_res") {
            const response = message.toString();
            resolve(response);
          }
        });
      });

      try {
        const response = await waitForResponse;
        // console.log(response);
        if (response) {
          //view time
          // console.log(formattedDate);
          await Attendances.create({
            Hourmin: gioPhut,
            Time: formattedDate,
            UserId: checkUser.id,
          });
        }
        return res.json({
          message: "Successfully",
          errCode: 0,
          data: response,
        });
      } catch (error) {
        return res.json({
          errCode: 1,
          message: "No MQTT response . Please check",
          data: error.message,
        });
      }
    }
  } catch (error) {
    return res.json({
      errCode: -1,
      message: "err server",
      data: error,
    });
  }
};
const getFullAttendance = async (req, res) => {
  const { UserId, Time } = req.body;
  if (!UserId || !Time) {
    return res.json({
      errCode: 1,
      message: "Dont enought information",
    });
  }
  try {
    const user = await Users.findOne({
      where: { id: UserId },
    });
    const data = await Attendances.findAll({
      where: {
        UserId: UserId,
        Time: Time,
      },
      include: [
        {
          model: Users,
          as: "user",
          attributes: ["userName", "userCode", "userImage"],
        },
      ],
    });
    return res.json({
      errCode: 0,
      message: "get full successfuly",
      length: data.length,
      img: user.userImage,
      code: user.userCode,
      name: user.userName,
      data: data,
    });
  } catch (error) {
    return res.json({
      errCode: -1,
      message: "err server",
      data: error,
    });
  }
};
module.exports = {
  createAttendance,
  getFullAttendance,
};
