import { useState } from "react";
import "./Attendance.css";
import axios from "axios";
import { URL } from "../../link_server";
import { ToastContainer, toast } from "react-toastify";
function Attendance() {
  const notify = (data) => toast(`${data}`);
  const [userCode, setUserCode] = useState("");
  const handleSendData = () => {
    if (!userCode) {
      alert("Please enter user Code !");
      return;
    }
     const data = {
       userCode : userCode
     };
     axios
       .post(`${URL}/attendance/create`, data, {
         withCredentials: true,
       })
       .then(function (response) {
         console.log(response);
         notify(response.data.message);
       })
       .catch(function (error) {
         console.log(error);
       });
  };

  return (
    <>
      <div className="attendance">
        <img
          className="attendance_img"
          alt=""
          src="https://helen.edu.vn/wp-content/uploads/2021/10/helen-attendance-management-solution.jpeg"
        />
        <div className="attendance_form">
          <h3 className="attendance_form_name">
            <i class="fa-solid fa-clipboard-user"></i> Attendance Form
          </h3>
          <div className="attendance_form_input">
            <h3>UserCode :</h3>{" "}
            <input
              value={userCode}
              placeholder="Ex : 20203520"
              onChange={(e) => {
                setUserCode(e.target.value);
              }}
            />
            <div onClick={handleSendData} className="attendance_form_send">
              SEND
            </div>
          </div>
          <p className="attendance_form_text">
            <i class="fa-solid fa-bullhorn" style={{ padding: "0px 10px" }}></i>
            An attendance tracking system offers a range of benefits for
            organizations and institutions. It simplifies the process of
            recording and managing attendance, providing a more efficient and
            accurate way to keep track of who is present and who is not
          </p>
        </div>
      </div>
      <ToastContainer />
    </>
  );
}

export default Attendance;
