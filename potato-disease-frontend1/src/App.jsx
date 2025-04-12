import React, { useState } from "react";


import { Carousel } from 'react-responsive-carousel';
import 'react-responsive-carousel/lib/styles/carousel.min.css'; // Required CSS



function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
  };

  const handleSubmit = async () => {
    if (!image) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("file", image);
    
    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });
      
      const data = await response.json();
      setResult(data);
    } catch (error) {
      alert("Error predicting. Is the backend running?");
    }
    setLoading(false);
  };
  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-400  p-6">
    {/* <div className="min-h-screen bg-gradient-to-br from-green-100 to-yellow-300 p-6"> */}
  
      {/* Header for Farmers */}
      <div className="bg-white shadow-lg rounded-2xl p-6 text-center">
        <h1 className="text-4xl font-bold text-green-800 mb-4">
          Welcome, Farmers! <span role="img" aria-label="seedling">🌱</span>
        </h1>
        <p className="text-gray-700 text-lg">
          This tool is designed to help you identify potential diseases in your potato crops
          by simply uploading a photo of a potato leaf. Our AI-powered system will analyze the image
          and provide quick, accurate results to support your farming decisions and crop health.
        </p>
      </div>
  
      {/* Carousel */}
      <div className="p-10 bg-gradient-to-br flex items-center justify-center">
        <div className="w-full max-w-4xl">
          <Carousel
            showThumbs={false}
            infiniteLoop
            autoPlay
            showStatus={false}
            showIndicators={true}
          >
            <div>
              <img
                src="https://images.yourstory.com/cs/5/f02aced0d86311e98e0865c1f0fe59a2/indian-farmer-1610471656527.png"
                alt="First"
                className="h-[600px] w-full object-cover rounded-xl"
              />
            </div>
            <div>
              <img
                src="https://cropaia.com/wp-content/uploads/Potato-field.jpg"
                alt="Second"
                className="h-[600px] w-full object-cover rounded-xl"
              />
            </div>
            <div>
              <img
                src="https://plantix.net/en/library/assets/custom/crop-images/potato.jpeg"
                alt="Third"
                className="h-[600px] w-full object-cover rounded-xl"
              />
            </div>
          </Carousel>
        </div>
      </div>
  
     


      {/* Disease Information */}
<div className="bg-white shadow-xl rounded-2xl p-10 mb-10">
  <h2 className="text-3xl font-bold text-green-700 text-center mb-6">
    Potato Leaf Diseases We Detect
  </h2>

  <div className="grid md:grid-cols-2 gap-10">
    {/* Early Blight */}
    <div className="text-center">
      <h3 className="text-2xl font-semibold text-green-600 mb-2">Early Blight</h3>
      <p className="text-gray-700 mb-4">
        Early blight is a common fungal disease caused by <i>Alternaria solani</i>. It appears as dark brown or black
        concentric spots on older leaves, eventually causing them to wither and drop. Early detection is crucial
        to prevent yield loss.
      </p>
      <img
        src="https://cdn.shortpixel.ai/spai/w_775+q_lossless+ret_img+to_webp/doraagri.com/wp-content/uploads/2019/09/early-blight.jpg"
        alt="Early Blight Leaf"
        className="w-full h-64 object-cover rounded-xl border mb-4"
      />
      <div className="bg-green-50 p-4 rounded-xl text-left text-gray-800 shadow-sm border border-green-200">
        <h4 className="text-lg font-semibold text-green-700 mb-2">What To Do:</h4>
        <ul className="list-disc list-inside space-y-1">
          <li>Remove and destroy infected leaves immediately.</li>
          <li>Avoid overhead irrigation to reduce leaf wetness.</li>
          <li>Apply fungicides like Chlorothalonil or Mancozeb early.</li>
          <li>Rotate crops and avoid planting potatoes in the same soil year after year.</li>
        </ul>
      </div>
    </div>

    {/* Late Blight */}
    <div className="text-center">
      <h3 className="text-2xl font-semibold text-green-600 mb-2">Late Blight</h3>
      <p className="text-gray-700 mb-4">
        Late blight is a severe disease caused by <i>Phytophthora infestans</i>, responsible for the Irish potato famine.
        It spreads rapidly in wet conditions, showing as water-soaked lesions that turn brown and spread across the leaf.
      </p>
      <img
        src="https://spudsmart.com/wp-content/uploads/2017/05/potato_late-blight_08_zoom-Photo-OMAFRA-900x580.jpeg.webp"
        alt="Late Blight Leaf"
        className="w-full h-64 object-cover rounded-xl border mb-4"
      />
      <div className="bg-green-50 p-4 rounded-xl text-left text-gray-800 shadow-sm border border-green-200">
        <h4 className="text-lg font-semibold text-green-700 mb-2">What To Do:</h4>
        <ul className="list-disc list-inside space-y-1">
          <li>Remove infected plants and avoid composting them.</li>
          <li>Use copper-based fungicides or systemic fungicides (e.g., Metalaxyl).</li>
          <li>Ensure proper spacing between plants for airflow.</li>
          <li>Monitor crops frequently, especially after rainy weather.</li>
        </ul>
      </div>
    </div>
  </div>
</div>

  


      <div className="bg-white shadow-2xl rounded-2xl p-10 max-w-4xl mx-auto mb-12 border-t-4 border-green-500">
        <h2 className="text-4xl text-green-800 font-extrabold mb-6 text-center tracking-wide">How It Works <span role="img" aria-label="tools">🛠️</span></h2>
        
        <div className="bg-green-50 rounded-xl p-6 space-y-4 text-lg text-gray-700 leading-relaxed">
          <div className="flex items-start space-x-3">
            <span className="text-green-700 font-bold">1.</span>
            <p>Click on the upload box and select a clear image of a potato leaf.</p>
          </div>
          <div className="flex items-start space-x-3">
            <span className="text-green-700 font-bold">2.</span>
            <p>Our AI model will process the image in a few seconds.</p>
          </div>
          <div className="flex items-start space-x-3">
            <span className="text-green-700 font-bold">3.</span>
            <p>You will receive a prediction along with a confidence score.</p>
          </div>
          <div className="flex items-start space-x-3">
            <span className="text-green-700 font-bold">4.</span>
            <p>Take necessary actions as advised to ensure healthy crop yield!</p>
          </div>
        </div>
      </div>





     
      <div className="flex justify-around  flex-wrap">
        <div className="bg-white shadow-xl rounded-2xl p-8 w-full max-w-lg mx-auto text-center mb-10 flex justify-center items-center">
          <div>
            <h2 className="text-3xl font-bold text-green-700 mb-4">
              <span role="img" aria-label="potato">🥔</span> Potato Disease Detector
            </h2>
            <p className="text-gray-600 mb-6">Upload a leaf image to check for disease</p>
  
            <input
              type="file"
              accept="image/*"
              onChange={handleImageChange}
              className="mb-4 w-full border rounded-lg p-2"
            />
  
            {preview && (
              <img
                src={preview}
                alt="Preview"
                className="mx-auto mb-4 max-h-64 object-contain rounded-lg border"
              />
            )}
  
            <button
              onClick={handleSubmit}
              disabled={loading}
              className="bg-green-600 hover:bg-green-700 text-white px-6 py-2 rounded-xl font-semibold transition duration-300"
            >
              {loading ? "Predicting..." : "Predict"}
            </button>
  
            {result && (
              <div className="mt-6 p-4 bg-green-50 rounded-xl shadow-inner">
                <p className="text-lg font-medium text-gray-700">
                  <span className="text-green-800 font-bold">Prediction:</span> {result.class}
                </p>
                <p className="text-md text-gray-600">
                  Confidence: {(result.confidence * 100).toFixed(2)}%
                </p>
              </div>
            )}
          </div>
        </div>
  
      </div>
      


      <footer className="bg-white mt-16 rounded-2xl shadow-md p-8 text-center text-green-800">
        <div className="flex justify-around">
          <div>
            
            <h3 className="text-2xl font-bold mb-4">Project Contributors</h3>
            <ul className="space-y-1 text-green-700 ">
              <li>PRAJJWAL SHUKLA (23BSA10197)</li>
              <li>AADYA SAHU (23BSA10016)</li>
              <li>TUSHAR SHARAD MATEY (23BSA10034)</li>
              <li>YASH RAKA (23BSA10036)</li>
              <li>BERA NEEL KIRANBHAI (23BSA10008)</li>
            </ul>
          </div>
          <div>

            <h4 className="text-9xl">Supervisor</h4>
            <p className="mb-4 text-[20px] text-green-700 ">Dr. Vijay Bircha</p>
          </div>
          <div >

            <h4 className="text-xl font-semibold mb-1">Reviewers</h4>
            <br />
            <p className="text-green-700">Reviewer 1: Dr. Hariharan R.</p>
            <p className="text-green-700">Reviewer 2: Dr. Bhupendra Panchal</p>
          </div>
        </div>

      <div className="mt-6 text-sm text-gray-500">© {new Date().getFullYear()} Potato Disease Detection Project.</div>
    </footer>

    </div>
  );
  
}
export default App;
//   return (
//     <div className="min-h-screen bg-gradient-to-br from-green-100 to-green-300 p-6">
        
  
//       {/* Header for Farmers */}
//       <div className="  bg-white shadow-lg rounded-2xl p-6 text-center">
//         <h1 className="text-4xl font-bold text-green-800 mb-4">Welcome, Farmers! <span role="img" aria-label="seedling">🌱</span>
//         </h1>
//         <p className="text-gray-700 text-lg">
//           This tool is designed to help you identify potential diseases in your potato crops
//           by simply uploading a photo of a potato leaf. Our AI-powered system will analyze the image
//           and provide quick, accurate results to support your farming decisions and crop health.
//         </p>
//       </div>

//     <div className="p-10 bg-gradient-to-br  flex items-center justify-center">
//       <div className="w-full max-w-4xl">
//         <Carousel
//           showThumbs={false}
//           infiniteLoop
//           autoPlay
//           showStatus={false}
//           showIndicators={true}
//         >
//           <div>
//             <img
//               src="https://images.yourstory.com/cs/5/f02aced0d86311e98e0865c1f0fe59a2/indian-farmer-1610471656527.png"
//               alt="First"
//               className="h-[600px] w-full object-cover rounded-xl"
//             />
//           </div>
//           <div>
//             <img
//               src="https://cropaia.com/wp-content/uploads/Potato-field.jpg"
//               alt="Second"
//               className="h-[600px] w-full object-cover rounded-xl"
//             />
//           </div>
//           <div>
//             <img
//               src="https://plantix.net/en/library/assets/custom/crop-images/potato.jpeg"
//               alt="Third"
//               className="h-[600px] w-full object-cover rounded-xl"
//             />
//           </div>
//         </Carousel>
//       </div>
//     </div>

//                     {/* How It Works */}
//                     <div className="bg-white shadow-md rounded-2xl p-8 max-w-4xl mx-auto mb-12">
//       <h2 className="text-3xl text-green-700 font-bold mb-4 text-center">How It Works 🛠️</h2>
//       <ol className="list-decimal list-inside text-gray-700 space-y-2">
//         <li>Click on the upload box and select a clear image of a potato leaf.</li>
//         <li>Our AI model will process the image in a few seconds.</li>
//         <li>You will receive a prediction with confidence level.</li>
//         <li>Take necessary steps as advised to maintain crop health!</li>
//       </ol>
//     </div>

     
//                 <div className="mx-auto bg-white shadow-xl rounded-2xl p-8 w-full max-w-lg  text-center mb-10 flex justify-center items-center">
//                   <div>
//                     <h2 className="text-3xl font-bold text-green-700 mb-4"><span role="img" aria-label="potato">🥔</span>
//                     Potato Disease Detector</h2>
//                     <p className="text-gray-600 mb-6">Upload a leaf image to check for disease</p>

//                     <input
//                       type="file"
//                       accept="image/*"
//                       onChange={handleImageChange}
//                       className="mb-4 w-full border rounded-lg p-2"
//                     />

//                     {preview && (
//                       <img
//                         src={preview}
//                         alt="Preview"
//                         className="mx-auto mb-4 max-h-64 object-contain rounded-lg border"
//                       />
//                     )}

//                     <button
//                       onClick={handleSubmit}
//                       disabled={loading}
//                       className="bg-green-600 hover:bg-green-700 text-white px-6 py-2 rounded-[50%] font-semibold transition duration-300"
//                     >
//                       {loading ? "Predicting..." : "Predict"}
//                     </button>

//                     {result && (
//                       <div className="mt-6 p-4 bg-green-50 rounded-xl shadow-inner">
//                         <p className="text-lg font-medium text-gray-700">
//                           <span className="text-green-800 font-bold">Prediction:</span> {result.class}
//                         </p>
//                         <p className="text-md text-gray-600">
//                           Confidence: {(result.confidence * 100).toFixed(2)}%
//                         </p>
//                       </div>
//                     )}
//                   </div>
//                 </div>
//                 <div className="mx-auto relative overflow-hidden bg-white/30 backdrop-blur-md shadow-2xl rounded-3xl p-8 w-full max-w-lg text-center mb-12 border border-green-400/50 ring-1 ring-green-300/30 hover:shadow-green-300 transition-all duration-500">

// <h2 className="text-3xl font-extrabold text-green-900 mb-6 animate-pulse">
//   🌟 Team Behind the Magic 🌟
// </h2>

// <div className="mb-6">
//   <h3 className="text-2xl font-semibold text-green-800 mb-1">🧑‍🏫 Supervisor</h3>
//   <p className="text-gray-800 text-lg">Dr. Vijay Bircha</p>
// </div>

// <div className="mb-6">
//   <h3 className="text-2xl font-semibold text-green-800 mb-1">🧪 Reviewers</h3>
//   <p className="text-gray-800 text-lg">Dr. Hariharan R.</p>
//   <p className="text-gray-800 text-lg">Dr. Bhupendra Panchal</p>
// </div>

// <div>
//   <h3 className="text-2xl font-bold text-green-800 mb-4">🚀 Contributors</h3>
//   <ul className="grid gap-3 text-left text-gray-900 font-semibold text-lg px-4">
//     <li className="hover:text-green-600 transition duration-300">🌱 PRAJJWAL SHUKLA (23BSA10197)</li>
//     <li className="hover:text-green-600 transition duration-300">🌱 AADYA SAHU (23BSA10016)</li>
//     <li className="hover:text-green-600 transition duration-300">🌱 TUSHAR SHARAD MATEY (23BSA10034)</li>
//     <li className="hover:text-green-600 transition duration-300">🌱 YASH RAKA (23BSA10036)</li>
//     <li className="hover:text-green-600 transition duration-300">🌱 BERA NEEL KIRANBHAI (23BSA10008)</li>
//   </ul>
// </div>

// {/* Optional glowing ring animation */}
// {/* <div className="absolute -inset-1 bg-gradient-to-r from-green-400 to-green-200 opacity-20 blur-lg rounded-3xl animate-pulse"></div> */}
// </div>
    
             
      

      

//     </div>
//   );



