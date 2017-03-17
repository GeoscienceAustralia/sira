/* tslint:disable:no-unused-variable */

import { TestBed, async, inject } from '@angular/core/testing';
import { ClassMetadataServiceService } from './class-metadata-service.service';

describe('ClassMetadataServiceService', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [ClassMetadataServiceService]
    });
  });

  it('should ...', inject([ClassMetadataServiceService], (service: ClassMetadataServiceService) => {
    expect(service).toBeTruthy();
  }));
});
