from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import date

class Role(BaseModel):
    """
    The Role class represents a specific role in the profile.
    """
    title: str
    company: Optional[str]
    start_date: Optional[date] = Field(alias="startDate")
    end_date: Optional[date] = Field(alias="endDate")
    years_in_role: Optional[float] = Field(alias="yearsInRole")
    location: Optional[str]

    def to_dict(self):
        """
        Convert Role instance to dictionary, converting date fields to strings.
        """
        role_dict = self.dict()
        if isinstance(self.start_date, date):
            role_dict['startDate'] = self.start_date.strftime('%Y-%m-%d')
        if isinstance(self.end_date, date):
            role_dict['endDate'] = self.end_date.strftime('%Y-%m-%d')
        return role_dict

class Profile(BaseModel):
    """
    The Profile class represents a user's career profile.
    """
    id: Optional[str] = Field(default=None, alias="_id")
    profile_id: str = Field(alias="profileId")
    path: str
    roles: List[Role]

    def to_dict(self):
        """
        Convert Profile instance to dictionary, including nested Role instances.
        """
        profile_dict = self.dict(by_alias=True)
        profile_dict['roles'] = [role.to_dict() for role in self.roles]
        return profile_dict

    class Config:
        """
        The Config inner class is used to configure the
        behavior of the Pydantic model.
        """
        populate_by_name = True

class ProfileList(BaseModel):
    """
    The ProfileList class represents a list of profiles.
    This class is used when deserializing a collection/array
    of profiles.
    """
    items: List[Profile]

    def to_dict_list(self):
        """
        Convert ProfileList to a list of dictionaries.
        """
        return [profile.to_dict() for profile in self.items]
