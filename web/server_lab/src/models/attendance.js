module.exports = (sequelize, DataTypes) => {
  const Attendances = sequelize.define("Attendances", {
    Time: {
      type: DataTypes.STRING,
      allowNull: false,
    },
    Hourmin: {
      type: DataTypes.STRING,
      allowNull: false,
    }
  });
  Attendances.associate = (models) => {
    Attendances.belongsTo(models.Users, { foreignKey: "UserId", as: "user" });
  };

  return Attendances;
};
