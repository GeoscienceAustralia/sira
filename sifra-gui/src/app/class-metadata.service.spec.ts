/* tslint:disable:no-unused-variable */

import { TestBed, async, inject } from '@angular/core/testing';
import { ClassMetadataService } from './class-metadata.service';

describe('ClassMetadataService', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [ClassMetadataService]
    });
  });

  it('should ...', inject([ClassMetadataService], (service: ClassMetadataService) => {
    expect(service).toBeTruthy();
  }));
});
