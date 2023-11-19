// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import {getStorage} from 'firebase/storage';
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyBy1gH65qewMUZa9aKMSyckM0xr6SPQDhU",
  authDomain: "imagepostcommentlike.firebaseapp.com",
  projectId: "imagepostcommentlike",
  storageBucket: "imagepostcommentlike.appspot.com",
  messagingSenderId: "37815532782",
  appId: "1:37815532782:web:07d3be7839a9a644ea07ff",
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

export const storage = getStorage(app);
