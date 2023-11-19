import "./Camera.css";
import Webcam from "react-webcam";
// eslint-disable-next-line no-unused-vars
import slugify from "slugify";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import axios from "axios";
import { useEffect, useState } from "react";
import { storage } from "../../firebase";
import { URL } from "../../link_server";
// import { v4 } from "uuid";
import { ref, uploadString, getDownloadURL } from "firebase/storage";
import Boxuser from "../../components/Boxuser/Boxuser";
function Camera() {
  const videoConstraints = {
    width: 1280,
    height: 1280,
    facingMode: "user",
  };
  const [checkSearch , setCheckSearch] = useState(false)
  const [f5, setF5] = useState(false);
  const [fix, setFix] = useState(false);
  const [dataUser, setDataUser] = useState({
    userName: "",
    userCode: "",
    userImage: "",
  });
  const [fullUser, setFullUser] = useState([]);
  const [search, setSearch] = useState("");
  const [userSearch, setUserSearch] = useState({});
  const [loadding, setLoadding] = useState(false);
  const notify = (data) => toast(`${data}`);
const handleKeyDown = (event) => {
  if (event.key === "Enter") {
    searchData();
  }
};
  const searchData = () => {
    const abc = fullUser.find((item) => {
      return item.userCode === search;
    });
    if (abc) {
      setUserSearch(abc);
      setCheckSearch(true);
    } else {
       setCheckSearch(false);
      return;
    }
  };

  const handleFixData = () => {
    if (!dataUser.userName) {
      alert("Please enter a name before press fix button");
      return;
    }
    const slug = slugify(dataUser.userName, {
      replacement: "_",
      lower: true,
    });
    setDataUser({ ...dataUser, userName: slug.toUpperCase() });
    setFix(true);
  };
  const handleSendData = () => {
    if (!dataUser.userCode || !dataUser.userName || !dataUser.userImage) {
      alert("ĐIỀN ĐỦ THÔNG TIN BẠN ƠI");
      return;
    }
    const data = {
      userName: dataUser.userName,
      userCode: dataUser.userCode,
      userImage: dataUser.userImage,
    };
    axios
      .post(`${URL}/user/register`, data, {
        withCredentials: true,
      })
      .then(function (response) {
        console.log(response);
        notify(response.data.message);
        if (response.data.errCode === 0) {
          setDataUser({ userName: "", userCode: "", userImage: "" });
          setFix(false);
          setF5(!f5);
        }
      })
      .catch(function (error) {
        console.log(error);
      });
  };

  useEffect(() => {
    axios
      .get(`${URL}/user/full`, {
        withCredentials: true,
      })
      .then(function (response) {
        if (response.data.data.length > 0) {
          setFullUser([...response.data.data]);
        }
      })
      .catch(function (error) {
        console.log(error);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [f5]);

  //box show
  const [showBox, setShowBox] = useState(false);

  const toggleBox = () => {
    setShowBox(!showBox);
  };
  return (
    <>
      <div className="webcam">
        <div className="webcam_left">
          <h3 className="webcam_left_rec">
            <i
              class="fa-solid fa-circle"
              style={{ color: "red", padding: "0px 5px" }}
            ></i>
            <span style={{ color: "#0a3871", fontSize: "25px" }}>REC</span>
          </h3>
          <Webcam
            audio={false}
            height={500}
            screenshotFormat="image/jpeg"
            width={500}
            videoConstraints={videoConstraints}
            className="webcam_cam"
          >
            {({ getScreenshot }) => (
              <div
                className="webcam_button"
                onClick={() => {
                  if (!dataUser.userCode || !dataUser.userName || !fix) {
                    alert(
                      "Please enter the name and code first . And Then press the Fix button !"
                    );
                    return;
                  }
                  setLoadding(true);
                  const imageSrc = getScreenshot();
                  const linkbase64 = imageSrc;
                  const newfileName = `${dataUser.userName}-${dataUser.userCode}.jpg`;
                  const storageRef = ref(storage, `images/${newfileName}`);
                  uploadString(storageRef, linkbase64, "data_url").then(
                    (snapshot) => {
                      getDownloadURL(storageRef)
                        .then((downloadURL) => {
                          setLoadding(false);
                          setDataUser({ ...dataUser, userImage: downloadURL });
                        })
                        .catch((error) => {
                          console.error("Error getting download URL:", error);
                        });
                    }
                  );
                }}
              >
                <i class="fa-solid fa-camera icon_camera"></i>
              </div>
            )}
          </Webcam>
        </div>
        <div className="webcam_right">
          <h3
            style={{ fontSize: "28px", textAlign: "center", padding: "10px" }}
          >
            <i class="fa-solid fa-align-left"></i> Create User
          </h3>
          <div className="webcam_right_content">
            <div className="webcam_right_content_cam">
              {dataUser.userImage ? (
                <img
                  alt=""
                  src={dataUser.userImage}
                  style={{ width: "100%", height: "100%" }}
                />
              ) : (
                <>
                  {loadding ? (
                    <>
                      <i class="fa-sharp fa-regular fa-loader fa-spin fa-spin-reverse">
                        <i
                          class="fa-solid fa-circle-notch"
                          style={{ color: "#0a3871", fontSize: "50px" }}
                        ></i>
                      </i>
                    </>
                  ) : (
                    <>
                      {" "}
                      <i class="fa-solid fa-image icon_image"></i>
                    </>
                  )}
                </>
              )}
            </div>
            <div className="webcam_right_content_input">
              <h3>
                User Name : <i class="fa-solid fa-signature"></i>
              </h3>
              <input
                value={dataUser.userName}
                placeholder="Ex : DO_VAN_NGOC"
                onChange={(e) => {
                  setDataUser({ ...dataUser, userName: e.target.value });
                }}
              />
              <h3>
                User Code : <i class="fa-solid fa-hashtag"></i>
              </h3>
              <input
                value={dataUser.userCode}
                placeholder="Ex : 20203520"
                onChange={(e) => {
                  setDataUser({ ...dataUser, userCode: e.target.value });
                }}
              />
              <div className="webcam_right_content_btn">
                <div className="btn_fixdata" onClick={handleFixData}>
                  FIX
                </div>
                <div className="btn_sendata" onClick={handleSendData}>
                  SEND
                </div>
              </div>
            </div>
          </div>
          <div className="webcam_right_footer">
            <h3>*Instruct (Step By Step)</h3>
            <h4>
              <i class="fa-solid fa-circle"></i> Step 1 : Enter information in
              the input box <i class="fa-solid fa-keyboard"></i> <br />
              <i class="fa-solid fa-circle"></i> Step 2 : check information{" "}
              <i class="fa-solid fa-check"></i> <br />
              <i class="fa-solid fa-circle"></i> Step 3 : Click the fix button{" "}
              <i class="fa-solid fa-tablet-button"></i> <br />
              <i class="fa-solid fa-circle"></i> Step 4 : Click on the icon
              camera <i class="fa-solid fa-camera"></i> <br />
              <i class="fa-solid fa-circle"></i> Step 5 : Click the send button{" "}
              <i class="fa-solid fa-share"></i>
            </h4>
          </div>
        </div>
      </div>

      <div className="btn_displaybox" onClick={toggleBox}>
        <i class="fa-solid fa-chevron-left"></i> List User
      </div>
      <div className={`slide-in ${showBox ? "show" : ""}`}>
        <div className="box-content">
          <div className="box_content_header">
            <i
              class="fa-solid fa-xmark"
              style={{
                fontSize: "40px",
                color: "#ddd",
                padding: "10px",
                cursor: "pointer",
              }}
              onClick={toggleBox}
            ></i>
            <div className="box_content_header_input">
              <i
                class="fa-solid fa-magnifying-glass"
                style={{ padding: "0px 10px", fontSize: "22px" }}
                onClick={searchData}
              ></i>
              <input
                className=""
                value={search}
                placeholder="Ex : 20203520"
                onChange={(e) => {
                  setSearch(e.target.value);
                }}
                onKeyDown={handleKeyDown}
              />

              {userSearch && checkSearch && (
                <div className="box_search">
                  <Boxuser
                    key={userSearch.id}
                    check={true}
                    infor={userSearch}
                  />
                  <div
                    className="box_search_close"
                    onClick={() => {
                      setSearch("");
                      setCheckSearch(false);
                    }}
                  >
                    <i class="fa-solid fa-xmark"></i>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="box_content_body">
            {fullUser &&
              fullUser.length > 0 &&
              fullUser.map((item, index) => {
                return <Boxuser key={index} check={true} infor={item} />;
              })}
          </div>
        </div>
      </div>
      <ToastContainer />
    </>
  );
}

export default Camera;
