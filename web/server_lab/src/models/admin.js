module.exports = (sequelize, DataTypes) => {
  const Admins = sequelize.define("Admins", {
    adminName: {
      type: DataTypes.STRING,
      allowNull: false,
    },
    adminPass: {
      type: DataTypes.STRING,
      allowNull: false,
    },
  });
  Admins.associate = (models) => {
    Admins.hasMany(models.Users, {
      onDelete: "cascade",
    }); 
  };
  return Admins;
};
