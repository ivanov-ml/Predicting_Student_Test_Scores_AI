from pydantic import BaseModel

class StudentData(BaseModel):
    study_hours: float
    class_attendance: float
    sleep_hours: float
    sleep_quality: int
    study_method: int
    facility_rating: int