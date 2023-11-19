import "./Adduser.css";
import axios from "axios";
import React, { useEffect, useState } from "react";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import slugify from "slugify";
import Modal from "react-modal";
import { storage } from "../../firebase";
import { v4 } from "uuid";
import { URL } from "../../link_server";
import { ref, uploadBytes, getDownloadURL } from "firebase/storage";
const customStyles = {
  content: {
    top: "50%",
    left: "50%",
    right: "auto",
    bottom: "auto",
    marginRight: "-50%",
    transform: "translate(-50%, -50%)",
    boxShadow: "rgba(3, 102, 214, 0.3) 0px 0px 0px 3px",
  },
};
function Adduser() {
  //upload image
  const [loadding, setLoadding] = useState(false);
  const uploadFile = (data) => {
    setLoadding(true);
    if (data == null) return;

    const imageRef = ref(storage, `images/${data.name + v4()}`);
    uploadBytes(imageRef, data).then((snapshot) => {
      getDownloadURL(snapshot.ref).then((url) => {
        setDataUser({ ...dataUser, userImg: url });
        setLoadding(false);
        setF5(!f5);
      });
    });
  };
  let subtitle;
  const [modalIsOpen, setIsOpen] = React.useState(false);
  function openModal() {
    setIsOpen(true);
  }
  function afterOpenModal() {
    subtitle.style.color = "#005eaa";
  }
  function closeModal() {
    setIsOpen(false);
  }
  const [f5, setF5] = useState(true);
  const [dataUser, setDataUser] = useState({
    userImg: "",
    userName: "",
    userCode: "",
  });
  const handleFixData = () => {
    const slug = slugify(dataUser.userName, {
      replacement: "_",
      lower: true,
    });
    setDataUser({ ...dataUser, userName: slug.toUpperCase() });
    setF5(!f5);
  };
  const notify = (data) => toast(`${data}`);
  const handleSendData = () => {
    if (!dataUser.userCode || !dataUser.userName || !dataUser.userImg) {
      alert("ĐIỀN ĐỦ THÔNG TIN BẠN ƠI");
      return;
    }
    console.log("data : ", dataUser);
    const data = {
      userName: dataUser.userName,
      userCode: dataUser.userCode,
      userImage: dataUser.userImg,
    };
    axios
      .post(`${URL}/user/register`, data, {
        withCredentials: true,
      })
      .then(function (response) {
        console.log(response);
        notify(response.data.message);
        if (response.data.errCode === 0) {
          setTimeout(closeModal, 2000);
        }
      })
      .catch(function (error) {
        console.log(error);
      });
  };
  useEffect(() => {}, [f5]);
  return (
    <>
      <div className="Adduser" onClick={openModal}>
        <i
          class="fa-solid fa-plus"
          style={{ paddingRight: "5px", marginTop: "3px" }}
        ></i>
        Add User
      </div>
      <Modal
        isOpen={modalIsOpen}
        onAfterOpen={afterOpenModal}
        onRequestClose={closeModal}
        style={customStyles}
        contentLabel="Example Modal"
      >
        <div
          ref={(_subtitle) => (subtitle = _subtitle)}
          style={{
            textAlign: "center",
            fontSize: "20px",
            marginBottom: "10px",
            color: "#005eaa",
            fontWeight: "600",
          }}
        >
          Add a user
        </div>
        <div className="modal_adduser">
          <label
            htmlFor="image_user"
            className="modal_adduser_img"
            style={{ cursor: "pointer" }}
          >
            <img alt="" src={dataUser.userImg} />
            {dataUser.userImg ? (
              <></>
            ) : (
              <>
                {loadding ? (
                  <i class="fa-sharp fa-regular fa-loader fa-spin fa-spin-reverse icon_camera_upload">
                    <i class="fa-solid fa-circle-notch"></i>
                  </i>
                ) : (
                  <i className="fa-solid fa-camera icon_camera_upload"></i>
                )}
              </>
            )}
          </label>
          <div className="modal_adduser_content">
            <h4>Name</h4>
            <input
              value={dataUser.userName}
              placeholder="EX : DO_VAN_NGOC"
              onChange={(e) => {
                setDataUser({ ...dataUser, userName: e.target.value });
              }}
            />
            <h4>Code</h4>
            <input
              value={dataUser.userCode}
              placeholder="EX : 20203520"
              onChange={(e) => {
                setDataUser({ ...dataUser, userCode: e.target.value });
              }}
            />
            <input
              type="file"
              id="image_user"
              name="fileInput"
              accept="image/*"
              hidden
              onChange={(e) => {
                uploadFile(e.target.files[0]);
              }}
            />
            <div className="modal_adduser_content_fix">
              <div className="btn_fix" onClick={handleFixData}>
                FIX
              </div>
              <div className="btn_send" onClick={handleSendData}>
                ADD
              </div>
            </div>
          </div>
        </div>
      </Modal>
      <ToastContainer />
    </>
  );
}

export default Adduser;
