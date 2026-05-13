using System;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Threading;
using UnityEngine;
using System.IO;
using System.Globalization;

public class VehicleAPIController : MonoBehaviour
{
    [Header("API Settings")]
    public int port = 8080;

    [Header("Camera Settings")]
    public Camera targetCamera;
    public int imageWidth = 640;
    public int imageHeight = 360;

    [Header("Vehicle Reference")]
    public DemoCarController carController;

    private Rigidbody carRigidbody;
    private float cachedSpeedKmh = 0f;
    private System.Reflection.FieldInfo velocityFieldInfo;

    private HttpListener listener;
    private Thread listenerThread;
    
    private RenderTexture captureRT;
    private Texture2D screenShot;
    private byte[] lastImageBytes;
    private object lockObject = new object();

    private void Start()
    {
        if (carController == null)
        {
            carController = FindObjectOfType<DemoCarController>();
        }

        // Rigidbody'yi aractan al (hiz okumak icin)
        if (carController != null)
        {
            velocityFieldInfo = typeof(DemoCarController).GetField("velocity", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            
            carRigidbody = carController.GetComponentInParent<Rigidbody>();
            if (carRigidbody == null)
                carRigidbody = carController.GetComponentInChildren<Rigidbody>();
        }
        if (carRigidbody == null)
        {
            carRigidbody = FindObjectOfType<Rigidbody>();
        }
        Debug.Log($"[VehicleAPI] Rigidbody found: {carRigidbody != null}");

        // Pre-allocate textures for capture (separate from camera's normal rendering)
        captureRT = new RenderTexture(imageWidth, imageHeight, 24);
        captureRT.antiAliasing = 1;
        screenShot = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);

        listener = new HttpListener();
        listener.Prefixes.Add($"http://localhost:{port}/");
        listener.Start();

        listenerThread = new Thread(HandleRequests);
        listenerThread.Start();
        
        Debug.Log($"[VehicleAPI] Server started on http://localhost:{port}/");
    }

    private void OnDestroy()
    {
        if (listener != null)
        {
            listener.Stop();
            listener.Close();
        }
        if (listenerThread != null)
        {
            listenerThread.Abort();
        }
        if (captureRT != null) captureRT.Release();
    }

    // Capture every frame in LateUpdate so it's always ready when requested.
    // This avoids the flickering caused by swapping targetTexture on-demand.
    private void LateUpdate()
    {
        // Hizi okumak icin once VolvoCars Velocity objesini dene, yoksa Rigidbody'ye dus
        if (velocityFieldInfo != null && carController != null)
        {
            var velObj = velocityFieldInfo.GetValue(carController) as VolvoCars.Data.Velocity;
            if (velObj != null)
            {
                cachedSpeedKmh = Mathf.Abs(velObj.Value) * 3.6f;
            }
        }
        else if (carRigidbody != null)
        {
            cachedSpeedKmh = carRigidbody.velocity.magnitude * 3.6f; // m/s -> km/h
        }
        CaptureCamera();
    }

    private void HandleRequests()
    {
        while (listener.IsListening)
        {
            try
            {
                var context = listener.GetContext();
                var request = context.Request;
                var response = context.Response;

                if (request.Url.AbsolutePath == "/control")
                {
                    string steerStr = request.QueryString["steer"];
                    string forwardStr = request.QueryString["forward"];
                    string parkStr = request.QueryString["park"];
                    
                    if (carController != null)
                    {
                        carController.useRemoteInput = true;
                        if (float.TryParse(steerStr, NumberStyles.Float, CultureInfo.InvariantCulture, out float s)) carController.remoteSteer = Mathf.Clamp(s, -1f, 1f);
                        if (float.TryParse(forwardStr, NumberStyles.Float, CultureInfo.InvariantCulture, out float f)) carController.remoteForward = Mathf.Clamp(f, -1f, 1f);
                        if (float.TryParse(parkStr, NumberStyles.Float, CultureInfo.InvariantCulture, out float p)) carController.remotePark = Mathf.Clamp(p, 0f, 1f);
                    }

                    byte[] buffer = System.Text.Encoding.UTF8.GetBytes("OK");
                    response.ContentLength64 = buffer.Length;
                    response.OutputStream.Write(buffer, 0, buffer.Length);
                }
                else if (request.Url.AbsolutePath == "/camera")
                {
                    lock (lockObject)
                    {
                        if (lastImageBytes != null)
                        {
                            response.ContentType = "image/jpeg";
                            response.ContentLength64 = lastImageBytes.Length;
                            response.OutputStream.Write(lastImageBytes, 0, lastImageBytes.Length);
                        }
                    }
                }
                else if (request.Url.AbsolutePath == "/status")
                {
                    // cachedSpeedKmh ana thread'de guncelleniyor, burada sadece okuyoruz
                    string json = $"{{\"speed_kmh\":{cachedSpeedKmh.ToString("F1", CultureInfo.InvariantCulture)}}}";
                    byte[] buffer = System.Text.Encoding.UTF8.GetBytes(json);
                    response.ContentType = "application/json";
                    response.ContentLength64 = buffer.Length;
                    response.OutputStream.Write(buffer, 0, buffer.Length);
                }
                response.Close();
            }
            catch (Exception) { }
        }
    }

    private void CaptureCamera()
    {
        if (targetCamera == null || captureRT == null || screenShot == null) return;

        // Save and restore state to avoid interfering with the camera's normal display
        RenderTexture previousCameraTarget = targetCamera.targetTexture;
        RenderTexture previousActive = RenderTexture.active;

        // Render to our offscreen RT
        targetCamera.targetTexture = captureRT;
        targetCamera.Render();

        // Restore the camera's original target IMMEDIATELY so the screen display is unaffected
        targetCamera.targetTexture = previousCameraTarget;

        // Read pixels from the offscreen RT
        RenderTexture.active = captureRT;
        screenShot.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        screenShot.Apply();
        RenderTexture.active = previousActive;
        
        lock (lockObject)
        {
            lastImageBytes = screenShot.EncodeToJPG(80);
        }
    }
}
