from sqlalchemy import Column, Integer, String, UniqueConstraint, DateTime, Float, Boolean, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

QUERY_BATCH = 200



class Study(Base):
    __tablename__ = "studies"
    study_id = Column(Integer, primary_key=True)
    study_name = Column(
        String, index=True, unique=True, nullable=False
    )

class Experiment(Base):
    __tablename__ = 'experiments'
    experiment_id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    start_time = Column(DateTime, nullable=False, default=func.now())
    end_time = Column(DateTime)
    study_id = Column(Integer, ForeignKey('studies.study_id', ondelete='CASCADE'))

class ExpMetadata(Base):
    __tablename__ = 'exp_metadata'
    exp_metadata_id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey('experiments.experiment_id', ondelete='CASCADE'), nullable=False)
    key = Column(String(255), nullable=False)
    value = Column(String(10240), nullable=False)

class Tag(Base):
    __tablename__ = 'tags'
    tag_id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)

class Component(Base):
    __tablename__ = 'components'
    component_id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)

class DataPoint(Base):
    __tablename__ = 'data_points'
    data_point_id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey('experiments.experiment_id', ondelete='CASCADE'), nullable=False)
    tag_id = Column(Integer, ForeignKey('tags.tag_id'), nullable=False)
    component_id = Column(Integer, ForeignKey('components.component_id'))
    int_value = Column(Integer)
    float_value = Column(Float)
    bool_value = Column(Boolean)
    string_value = Column(String(10240))
    step = Column(Integer)
    timestamp = Column(DateTime, nullable=False, default=func.now())
    __table_args__ = (
        UniqueConstraint('experiment_id', 'tag_id', 'component_id', 'step', name='uix_exp_tag_comp_id_step'),
    )

class Evaluation(Base):
    __tablename__ = 'evaluations'
    evaluation_id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey('experiments.experiment_id', ondelete='CASCADE'), nullable=False)
    init_position = Column(String(1020))
    goal_value = Column(String(510))
    reward = Column(Float)
    success = Column(Boolean)
    step = Column(Integer)
