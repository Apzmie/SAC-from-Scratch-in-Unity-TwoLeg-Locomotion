using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class ModelAgent : Agent
{  
    private float stiffness = 40f;
    private float damping = 20f;
    private float forceLimit = 80f;
    private ArticulationBody aBody;
    private ArticulationBody[] joints;
    private float[] startJointPositions;
    private float[] lowerLimits;
    private float[] upperLimits;
    private Vector3 startPos;
    private Quaternion startRot;
    private float Tanh(float x) => (float)System.Math.Tanh(x);
    private Vector3 lastPosition;
    private float[] lastActions;

    public override void Initialize()
    {
        aBody = GetComponent<ArticulationBody>();
        joints = GetComponentsInChildren<ArticulationBody>();
        joints = System.Array.FindAll(joints, j => j != aBody);
        startJointPositions = new float[joints.Length];
        lowerLimits = new float[joints.Length];
        upperLimits = new float[joints.Length];
        for (int i = 0; i < joints.Length; i++)
        {
            var joint = joints[i];
            var drive = joint.xDrive;

            startJointPositions[i] = joints[i].jointPosition[0] * Mathf.Rad2Deg;

            lowerLimits[i] = drive.lowerLimit;
            upperLimits[i] = drive.upperLimit;

            drive.stiffness = stiffness;
            drive.damping = damping;
            drive.forceLimit = forceLimit;
            drive.targetVelocity = 0f;
            joint.xDrive = drive;
        }

        startPos = transform.position;
        startRot = transform.rotation;
        lastActions = new float[joints.Length];
    }

    public override void OnEpisodeBegin()
    {
        aBody.TeleportRoot(startPos, startRot);
        aBody.linearVelocity = Vector3.zero;
        aBody.angularVelocity = Vector3.zero;

        for (int i = 0; i < joints.Length; i++)
        {
            joints[i].jointPosition = new ArticulationReducedSpace(startJointPositions[i] * Mathf.Deg2Rad);
            joints[i].jointVelocity = new ArticulationReducedSpace(0f);

            var drive = joints[i].xDrive;
            drive.target = startJointPositions[i];
            drive.targetVelocity = 0f;
            joints[i].xDrive = drive;
        }

        Physics.SyncTransforms();
        lastPosition = transform.localPosition;
        System.Array.Clear(lastActions, 0, lastActions.Length);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        Vector3 localVel = transform.InverseTransformDirection(aBody.linearVelocity);
        sensor.AddObservation(Tanh(localVel.x / 2f));
        sensor.AddObservation(Tanh(localVel.y / 2f));
        sensor.AddObservation(Tanh(localVel.z / 2f));
        Vector3 localAngVel = transform.InverseTransformDirection(aBody.angularVelocity);
        sensor.AddObservation(Tanh(localAngVel.x / 2f));
        sensor.AddObservation(Tanh(localAngVel.y / 2f));
        sensor.AddObservation(Tanh(localAngVel.z / 2f));
        Vector3 upVector = transform.InverseTransformDirection(Vector3.up);
        sensor.AddObservation(upVector);

        for (int i = 0; i < joints.Length; i++)
        {
            float currentPos = joints[i].jointPosition[0] * Mathf.Rad2Deg;
            float normalizedPos = Mathf.InverseLerp(lowerLimits[i], upperLimits[i], currentPos) * 2f - 1f;
            sensor.AddObservation(normalizedPos);

            float currentVel = joints[i].jointVelocity[0];
            sensor.AddObservation(Tanh(currentVel / 2f));
        }
        
        for (int i = 0; i < lastActions.Length; i++)
        {
            sensor.AddObservation(lastActions[i]);
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        for (int i = 0; i < joints.Length; i++)
        {
            float action = actions.ContinuousActions[i];
            lastActions[i] = action;
            float targetAngle = Mathf.Lerp(lowerLimits[i], upperLimits[i], (action + 1f) / 2f);

            var drive = joints[i].xDrive;
            drive.target = targetAngle;
            joints[i].xDrive = drive;
        }
        
        float upright = Vector3.Dot(transform.up, Vector3.up);        
        //AddReward(upright * 0.01f);       
        
        float forwardSpeed = Vector3.Dot(aBody.linearVelocity, transform.forward); 
        AddReward(forwardSpeed * 0.01f);
        
        
        if (upright < 0.3f)
        {
            SetReward(-1.0f);
            EndEpisode();
        }

        if (transform.localPosition.y < 1.0f)
        {
            SetReward(-1.0f);
            EndEpisode();
        }
    }
}
