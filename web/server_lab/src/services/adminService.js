const { Admins } = require("../models");
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");
const createAdmin = async (req, res) => {
  const { adminName, adminPass } = req.body;
  if (!adminName || !adminPass) {
    return res.json({
      errCode: 1,
      message: "not enought information",
      data: null,
    });
  }
  try {
    const checkAdmin = await Admins.findOne({
      where: { adminName: adminName },
    });
    if (checkAdmin) {
      return res.json({
        errCode: 1,
        message: "account already exists",
        data: null,
      });
    } else {
      const hash = bcrypt.hashSync(adminPass, 12);
      await Admins.create({
        adminName: adminName,
        adminPass: hash,
      });

      return res.json({
        errCode: 0,
        message: "create account success",
      });
    }
  } catch (error) {
    return res.json({
      errCode: -1,
      message: "err server",
      data: null,
    });
  }
};
const loginAdmin = async (req, res) => {
  const { adminName, adminPass } = req.body;
  if (!adminName || !adminPass) {
    return res.json({
      errCode: 1,
      message: "not enought information",
      data: null,
    });
  }
  try {
    const checkAdmin = await Admins.findOne({
      where: { adminName: adminName },
    });
    if (!checkAdmin) {
      return res.json({
        errCode: 1,
        message: "account not found",
        data: null,
      });
    } else {
      const checkPassword = bcrypt.compareSync(adminPass, checkAdmin.adminPass);

      if (!checkPassword) {
        return res.json({
          errCode: 1,
          message: "wrong password",
          data: null,
        });
      } else {
        const token = jwt.sign({ admin: checkAdmin }, "MYKEY", {
          expiresIn: "24h",
        });
        return res.json({
          errCode: 0,
          message: "login success",
          token: token,
        });
      }
    }
  } catch (error) {
    return res.json({
      errCode: -1,
      message: "err server",
      data: null,
    });
  }
};

module.exports = {
  createAdmin: createAdmin,
  loginAdmin: loginAdmin,
};
