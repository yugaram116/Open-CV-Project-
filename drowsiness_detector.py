import cv2, numpy as np, mediapipe as mp, time, pygame, sys
from scipy.spatial import distance as dist
from collections import deque

EAR_THRESHOLD=0.22
MAR_THRESHOLD=0.65
EAR_CONSEC_FRAMES=20
MAR_CONSEC_FRAMES=15
PERCLOS_WINDOW=300
PERCLOS_THRESHOLD=0.30
LEFT_EYE=[362,385,387,263,373,380]
RIGHT_EYE=[33,160,158,133,153,144]
MOUTH=[61,291,39,181,0,17,269,405]

def calc_ear(lm,idx,w,h):
    p=[(int(lm[i].x*w),int(lm[i].y*h)) for i in idx]
    A=dist.euclidean(p[1],p[5])
    B=dist.euclidean(p[2],p[4])
    C=dist.euclidean(p[0],p[3])
    return (A+B)/(2.0*C) if C else 0.0

def calc_mar(lm,idx,w,h):
    p=[(int(lm[i].x*w),int(lm[i].y*h)) for i in idx]
    A=dist.euclidean(p[2],p[6])
    B=dist.euclidean(p[3],p[5])
    C=dist.euclidean(p[0],p[1])
    return (A+B)/(2.0*C) if C else 0.0

try:
    pygame.mixer.init(frequency=44100,size=-16,channels=2,buffer=512)
    sr=44100
    t=np.linspace(0,0.4,int(sr*0.4),False)
    wave=(np.sin(2*np.pi*880*t)*32767).astype(np.int16)
    sound=pygame.sndarray.make_sound(np.column_stack([wave,wave]))
    audio_ok=True
    print("[Sound] Audio enabled.")
except:
    audio_ok=False
    print("[Sound] No audio, visual alerts only.")

mp_fm=mp.solutions.face_mesh
fm=mp_fm.FaceMesh(max_num_faces=1,refine_landmarks=True,
                  min_detection_confidence=0.5,min_tracking_confidence=0.5)
cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
if not cap.isOpened():
    print("[ERROR] Cannot open camera!")
    sys.exit(1)

ec=mc=dc=yc=tf=0
pb=deque(maxlen=PERCLOS_WINDOW)
last_alert=0
st=time.time()
print("[INFO] Camera opened! Press Q to quit.")

while True:
    ret,frame=cap.read()
    if not ret:
        print("[WARN] Cannot read camera frame.")
        break
    h,w=frame.shape[:2]
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    res=fm.process(rgb)
    EAR=0.0
    MAR=0.0
    al="NO FACE"
    tf+=1

    if res.multi_face_landmarks:
        lm=res.multi_face_landmarks[0].landmark
        EAR=(calc_ear(lm,LEFT_EYE,w,h)+calc_ear(lm,RIGHT_EYE,w,h))/2.0
        MAR=calc_mar(lm,MOUTH,w,h)

        for idx in [LEFT_EYE,RIGHT_EYE]:
            pts=np.array([(int(lm[i].x*w),int(lm[i].y*h)) for i in idx],np.int32)
            col=(0,50,255) if EAR<EAR_THRESHOLD else (0,220,100)
            cv2.drawContours(frame,[cv2.convexHull(pts)],-1,col,1)
        mpts=np.array([(int(lm[i].x*w),int(lm[i].y*h)) for i in MOUTH],np.int32)
        cv2.drawContours(frame,[cv2.convexHull(mpts)],-1,(0,165,255),1)

        pb.append(1 if EAR<EAR_THRESHOLD else 0)
        pc=float(np.mean(pb))
        al="AWAKE"

        if EAR<EAR_THRESHOLD:
            ec+=1
        else:
            ec=0

        if MAR>MAR_THRESHOLD:
            mc+=1
        else:
            if mc>=MAR_CONSEC_FRAMES:
                yc+=1
                print(f"[YAWN] #{yc}")
            mc=0

        if mc>=MAR_CONSEC_FRAMES:
            al="YAWNING"
            cv2.rectangle(frame,(0,h-80),(w,h),(0,100,180),-1)
            cv2.putText(frame,"YAWNING DETECTED",(w//2-160,h-25),
                        cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),3)
            if time.time()-last_alert>3:
                if audio_ok: sound.play()
                last_alert=time.time()

        if ec>=EAR_CONSEC_FRAMES or pc>PERCLOS_THRESHOLD:
            al="DROWSY"
            cv2.rectangle(frame,(0,h-80),(w,h),(0,0,200),-1)
            cv2.putText(frame,"DROWSINESS ALERT! WAKE UP!",(w//2-220,h-25),
                        cv2.FONT_HERSHEY_SIMPLEX,1.1,(255,255,255),3)
            if time.time()-last_alert>3:
                if audio_ok: sound.play()
                last_alert=time.time()
            if ec>=EAR_CONSEC_FRAMES:
                dc+=1
                print(f"[ALERT] Drowsy event #{dc}")
                ec=0
        elif ec>=EAR_CONSEC_FRAMES//2:
            al="WARNING"
    else:
        pc=float(np.mean(pb)) if pb else 0.0

    COLORS={"AWAKE":(0,220,100),"WARNING":(0,200,255),
            "DROWSY":(0,50,255),"YAWNING":(0,165,255),"NO FACE":(150,150,150)}
    ov=frame.copy()
    cv2.rectangle(ov,(0,0),(w,110),(15,15,15),-1)
    cv2.addWeighted(ov,0.65,frame,0.35,0,frame)
    cv2.putText(frame,f"EAR : {EAR:.3f}",(12,28),cv2.FONT_HERSHEY_SIMPLEX,0.65,(200,200,200),2)
    cv2.putText(frame,f"MAR : {MAR:.3f}",(12,55),cv2.FONT_HERSHEY_SIMPLEX,0.65,(200,200,200),2)
    cv2.putText(frame,f"PERCLOS: {pc*100:.1f}%",(12,82),cv2.FONT_HERSHEY_SIMPLEX,0.65,(200,200,200),2)
    cv2.putText(frame,al,(w-210,55),cv2.FONT_HERSHEY_SIMPLEX,1.1,COLORS.get(al,(255,255,255)),3)
    m2,s2=divmod(int(time.time()-st),60)
    cv2.putText(frame,f"Session {m2:02d}:{s2:02d}  Drowsy:{dc}  Yawns:{yc}",
                (10,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(140,140,140),1)
    cv2.imshow("Driver Drowsiness Detection | Q=quit",frame)
    if cv2.waitKey(1)&0xFF in [ord("q"),27]:
        break

cap.release()
cv2.destroyAllWindows()
fm.close()
print(f"Session ended. Drowsy alerts: {dc}, Yawns: {yc}")