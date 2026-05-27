using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents.Sensors;

public class RayVisualizer : MonoBehaviour
{
    [Header("Line Settings")]
    public float lineWidth = 0.02f;

    [Header("Colors")]
    public Color hitColor = Color.red;
    public Color missColor = Color.white;

    private RayPerceptionSensorComponent3D sensor;
    private List<LineRenderer> lineRenderers = new();

    private Material sharedMat;

    void Start()
    {
        sensor = GetComponent<RayPerceptionSensorComponent3D>();

        if (sensor == null)
        {
            Debug.LogError("RayPerceptionSensorComponent3D not found!");
            return;
        }

        sharedMat = new Material(Shader.Find("Sprites/Default"));

        var input = sensor.GetRayPerceptionInput();

        for (int i = 0; i < input.Angles.Count; i++)
        {
            GameObject rayObj = new GameObject($"Ray_{i}");
            rayObj.transform.SetParent(transform);

            LineRenderer lr = rayObj.AddComponent<LineRenderer>();

            lr.positionCount = 2;
            lr.startWidth = lineWidth;
            lr.endWidth = lineWidth;

            lr.useWorldSpace = true;

            lr.material = sharedMat;

            lr.startColor = missColor;
            lr.endColor = missColor;

            lineRenderers.Add(lr);
        }
    }

    void FixedUpdate()
    {
        if (sensor == null)
            return;

        var input = sensor.GetRayPerceptionInput();

        var output = RayPerceptionSensor.Perceive(input, false);

        for (int i = 0; i < output.RayOutputs.Length; i++)
        {
            var ray = output.RayOutputs[i];

            LineRenderer lr = lineRenderers[i];

            Vector3 startPos = ray.StartPositionWorld;

            Vector3 fullEndPos = ray.EndPositionWorld;

            Vector3 endPos;

            if (ray.HasHit)
            {
                endPos = Vector3.Lerp(
                    startPos,
                    fullEndPos,
                    ray.HitFraction
                );
            }
            else
            {
                endPos = fullEndPos;
            }

            lr.SetPosition(0, startPos);
            lr.SetPosition(1, endPos);

            Color lineColor = ray.HasHit ? hitColor : missColor;

            lr.startColor = lineColor;
            lr.endColor = lineColor;
        }
    }
}
