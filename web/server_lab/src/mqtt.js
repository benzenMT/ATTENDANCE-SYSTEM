const mqtt = require("mqtt");
 let client = mqtt.connect("https://test.mosquitto.org:1883");
module.exports = {
    client
}

