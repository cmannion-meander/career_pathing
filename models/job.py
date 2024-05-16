from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime

class Job(BaseModel):
    """
    The Job class represents a job in the dataset.
    """
    id: Optional[str] = Field(default=None, alias="_id")
    role_id: str = Field(alias="roleId")
    role_name: str = Field(alias="roleName")
    title: str
    company_name: Optional[str] = Field(alias="companyName")
    created_date: datetime = Field(alias="createdDate")
    description: str
    job_location: Optional[str] = Field(alias="jobLocation")

    class Config:
        """
        The Config inner class is used to configure the
        behavior of the Pydantic model.
        """
        populate_by_name = True

class JobList(BaseModel):
    """
    The JobList class represents a list of jobs.
    This class is used when deserializing a collection/array
    of jobs.
    """
    items: list[Job]
