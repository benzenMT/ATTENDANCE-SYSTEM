const mqtt = require("mqtt");
let client = mqtt.connect("https://test.mosquitto.org:1883");

client.on("connect", () => {
  setInterval(() => {
    client.subscribe("timekeep/new_employee");
    client.subscribe("time_keeping");
  }, 1000);
});
client.on("message", (topic, data) => {
  console.log(topic.toString());
  console.log(data.toString());
  if (topic === "time_keeping") {
    client.publish("time_keeping_res", "20203520");
    console.log("connected");
  }
  if (topic === "timekeep/new_employee") {
    client.publish("timekeep/new_employee_res", "20203520");
    console.log("connected");
  }
});
