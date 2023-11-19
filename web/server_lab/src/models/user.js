module.exports = (sequelize, DataTypes) => {
  const Users = sequelize.define("Users", {
    userName: {
      type: DataTypes.STRING,
      allowNull: false,
    },
    userCode: {
      type: DataTypes.STRING,
      allowNull: false,
    },
    userImage: {
      type: DataTypes.STRING,
      allowNull: false,
    },
  });
  Users.associate = (models) => {
    Users.hasMany(models.Attendances, {
      onDelete: "cascade",
    });
    Users.belongsTo(models.Admins, {
      foreignKey: "AdminId",
      as: "admin",
    });
  };

  return Users;
};
