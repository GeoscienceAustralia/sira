import { Component, Input, Output, EventEmitter } from '@angular/core';

@Component({
    selector: 'dict-display',
    template: `
        <div>
            <div>
                {{name}}
            </div>
            <element-chooser
                (publish)="doPublish($event)">
            </element-chooser>
        </div>
    `
})
export class DictDisplayComponent {
    @Input() name: string;

    @Output() publish = new EventEmitter();

    className: string = null;

    doPublish($event) {
        this.publish.emit({name: this.name, value: $event});
    }
}
