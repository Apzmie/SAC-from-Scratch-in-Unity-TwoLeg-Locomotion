using UnityEngine;
using System.Collections.Generic;

public class CustomRaySensor : MonoBehaviour
{
    [Header("Ray Settings")]
    public int rayCount;
    public float maxDistance;
    public float fieldOfView;

    [Header("Ray Origin Offset")]
    public Vector3 rayOffset;

    [Header("Detection")]
    public LayerMask detectableLayers;
    public string[] detectableTags = { };

    [Header("Scene + Play View")]
    public bool showRays = true;

    [Header("Line Renderer Materials")]
    public Material defaultMaterial;
    public Material hitMaterial;

    [Header("Line Renderer")]
    public float lineWidth = 0.02f;

    private LineRenderer[] lineRenderers;

    private Vector3 RayOrigin =>
        transform.position +
        transform.right * rayOffset.x +
        transform.up * rayOffset.y +
        transform.forward * rayOffset.z;

    private void Awake()
    {
        lineRenderers = new LineRenderer[rayCount];

        for (int i = 0; i < rayCount; i++)
        {
            GameObject lrObj = new GameObject($"Ray_{i}");
            lrObj.transform.parent = transform;

            LineRenderer lr = lrObj.AddComponent<LineRenderer>();

            lr.material = defaultMaterial;
            lr.startWidth = lineWidth;
            lr.endWidth = lineWidth;
            lr.positionCount = 2;
            lr.useWorldSpace = true;

            lineRenderers[i] = lr;
        }

        ApplyVisibility(showRays);
    }

    private void Update()
    {
        ApplyVisibility(showRays);

        UpdateRays();
    }

    private void ApplyVisibility(bool visible)
    {
        if (lineRenderers == null) return;

        for (int i = 0; i < lineRenderers.Length; i++)
        {
            if (lineRenderers[i] != null)
                lineRenderers[i].enabled = visible;
        }
    }

    private void UpdateRays()
    {
        Vector3 origin = RayOrigin;

        float angleStep = rayCount > 1
            ? fieldOfView / (rayCount - 1)
            : 0f;

        float startAngle = -fieldOfView / 2f;

        for (int i = 0; i < rayCount; i++)
        {
            float angle = startAngle + angleStep * i;

            Vector3 direction =
                Quaternion.Euler(0, angle, 0) * transform.forward;

            Vector3 endPoint;

            if (Physics.Raycast(origin, direction, out RaycastHit hit, maxDistance, detectableLayers))
            {
                endPoint = hit.point;

                if (lineRenderers[i] != null)
                    lineRenderers[i].material = hitMaterial;
            }
            else
            {
                endPoint = origin + direction * maxDistance;

                if (lineRenderers[i] != null)
                    lineRenderers[i].material = defaultMaterial;
            }

            if (lineRenderers[i] != null)
            {
                lineRenderers[i].SetPosition(0, origin);
                lineRenderers[i].SetPosition(1, endPoint);
            }
        }
    }

    public float[] GetObservations()
    {
        List<float> observations = new();

        Vector3 origin = RayOrigin;

        float angleStep = rayCount > 1
            ? fieldOfView / (rayCount - 1)
            : 0f;

        float startAngle = -fieldOfView / 2f;

        for (int i = 0; i < rayCount; i++)
        {
            float angle = startAngle + angleStep * i;

            Vector3 direction =
                Quaternion.Euler(0, angle, 0) * transform.forward;

            if (Physics.Raycast(origin, direction, out RaycastHit hit, maxDistance, detectableLayers))
            {
                observations.Add(hit.distance / maxDistance);

                foreach (string tag in detectableTags)
                    observations.Add(hit.collider.CompareTag(tag) ? 1f : 0f);
            }
            else
            {
                observations.Add(1f);

                for (int j = 0; j < detectableTags.Length; j++)
                    observations.Add(0f);
            }
        }

        return observations.ToArray();
    }

    private void OnDrawGizmos()
    {
        if (!showRays)
            return;

        Vector3 origin = RayOrigin;

        float angleStep = rayCount > 1
            ? fieldOfView / (rayCount - 1)
            : 0f;

        float startAngle = -fieldOfView / 2f;

        for (int i = 0; i < rayCount; i++)
        {
            float angle = startAngle + angleStep * i;

            Vector3 direction =
                Quaternion.Euler(0, angle, 0) * transform.forward;

            if (Physics.Raycast(origin, direction, out RaycastHit hit, maxDistance, detectableLayers))
            {
                Gizmos.color = Color.red;
                Gizmos.DrawLine(origin, hit.point);
            }
            else
            {
                Gizmos.color = Color.green;
                Gizmos.DrawLine(origin, origin + direction * maxDistance);
            }
        }
    }
}
