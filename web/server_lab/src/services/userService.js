const { Users } = require("../models");

const client = require("../mqtt");

client.client.subscribe("timekeep/new_employee_res");

const createUser = async (req, res) => {
  const { userName, userCode, userImage } = req.body;
  if (!userName || !userCode || !userImage) {
    return res.json({
      errCode: 1,
      message: "Dont enought information",
    });
  }
  try {
    const checkUser = await Users.findOne({
      where: { userCode: userCode },
    });
    if (checkUser) {
      return res.json({
        errCode: 1,
        message: "user already exists",
      });
    } else {
      // Tạo một promise để đợi phản hồi
      const waitForResponse = new Promise((resolve, reject) => {
        client.client.publish("timekeep/new_employee", `${userImage}`);
        setTimeout(() => {
          reject(new Error("Timeout waiting for response"));
        }, 5000); // Timeout sau 5 giây

        client.client.on("message", (topic, message) => {
          if (topic === "timekeep/new_employee_res") {
            const response = message.toString();
            resolve(response);
          }
        });
      });

      try {
        const response = await waitForResponse;
        if (response) {
          await Users.create({
            userName,
            userCode,
            userImage,
            AdminId: req.AdminId,
          });
        }
        return res.json({
          message: "Received response and created new user",
          data: response,
          errCode: 0,
        });
      } catch (error) {
        return res.json({
          errCode: 1,
          message: "No MQTT response . Please check",
          data: error.message,
        });
      }

      //   await Users.create({
      //     userName,
      //     userCode,
      //     userImage,
      //     AdminId: req.AdminId,
      //   });

      //   return res.json({
      //     errCode: 0,
      //     message: "create user success",
      //   });
    }
  } catch (error) {
    return res.json({
      errCode: -1,
      message: "err server",
      data: error,
    });
  }
};

const getFullUser = async (req, res) => {
  try {
    const fulluser = await Users.findAll({
      where: { AdminId: req.AdminId },
    });

    if (!fulluser) {
      return res.json({
        errCode: 1,
        message: "Dont find users",
        data: [],
      });
    } else {
      return res.json({
        errCode: 0,
        message: "find success",
        data: fulluser,
      });
    }
  } catch (error) {
    return res.json({
      errCode: -1,
      message: "Err Server",
      data: error,
    });
  }
};

module.exports = {
  createUser,
  getFullUser,
};
